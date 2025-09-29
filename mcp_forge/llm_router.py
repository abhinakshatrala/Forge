"""
LLM Router with stage-specific routing and difficulty-aware selection
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of LLM routing"""
    model: str
    endpoint: str
    response: Optional[str] = None
    latency_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


class LLMRequest(BaseModel):
    """LLM request structure"""
    prompt: str
    task: str
    complexity: int = 5
    profile: str = "pm"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    structured_output: bool = False


class LLMRouter:
    """
    LLM Router with stage-specific routing policies
    Supports local endpoints and difficulty-aware routing
    """
    
    def __init__(self, config):
        self.config = config
        self.providers = config.llm_providers
        self.profiles = config.profiles
        
        # Track usage statistics
        self.usage_stats = {}
        self.model_health = {}
        
    async def route_request(self, task: str, complexity: int, profile: str, prompt: str, **kwargs) -> RouteResult:
        """
        Route LLM request based on task, complexity, and profile
        """
        try:
            # Get profile-specific router config
            profile_config = self.profiles.get(profile, {})
            router_config = profile_config.get("llm_router", {})
            
            # Find matching route
            selected_model = self._select_model(task, complexity, router_config)
            
            # Get provider endpoint
            provider = self.providers.get(selected_model)
            if not provider:
                raise ValueError(f"Provider not found for model: {selected_model}")
                
            # Prepare request
            llm_request = LLMRequest(
                prompt=prompt,
                task=task,
                complexity=complexity,
                profile=profile,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                structured_output=kwargs.get("structured_output", False)
            )
            
            # Execute request
            start_time = datetime.now()
            response = await self._execute_request(provider, llm_request)
            end_time = datetime.now()
            
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update usage stats
            self._update_usage_stats(selected_model, latency_ms, len(prompt))
            
            return RouteResult(
                model=selected_model,
                endpoint=provider.endpoint,
                response=response,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            
            # Try fallback models
            fallback_models = router_config.get("fallbacks", ["local-small"])
            for fallback_model in fallback_models:
                try:
                    provider = self.providers.get(fallback_model)
                    if provider:
                        response = await self._execute_request(provider, llm_request)
                        return RouteResult(
                            model=fallback_model,
                            endpoint=provider.endpoint,
                            response=response
                        )
                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} failed: {fallback_error}")
                    continue
                    
            raise Exception(f"All models failed for request: {str(e)}")
            
    def _select_model(self, task: str, complexity: int, router_config: Dict[str, Any]) -> str:
        """
        Select appropriate model based on routing rules
        """
        routes = router_config.get("routes", [])
        default_model = router_config.get("default_model", "local-small")
        policy = router_config.get("policy", "balanced")
        
        # Find matching routes
        matching_routes = []
        for route in routes:
            match_criteria = route.get("match", {})
            
            # Check task match
            if "task" in match_criteria and match_criteria["task"] != task:
                continue
                
            # Check complexity range
            if "complexity_min" in match_criteria and complexity < match_criteria["complexity_min"]:
                continue
            if "complexity_max" in match_criteria and complexity > match_criteria["complexity_max"]:
                continue
                
            matching_routes.append(route)
            
        if not matching_routes:
            return default_model
            
        # Apply routing policy
        if policy == "cost_then_quality":
            # Prefer cheaper models first, then quality
            return self._select_by_cost(matching_routes)
        elif policy == "quality_then_latency":
            # Prefer higher quality models, then lower latency
            return self._select_by_quality(matching_routes)
        elif policy == "latency_then_cost":
            # Prefer lower latency, then cost
            return self._select_by_latency(matching_routes)
        else:
            # Balanced approach
            return self._select_balanced(matching_routes)
            
    def _select_by_cost(self, routes: List[Dict[str, Any]]) -> str:
        """Select model prioritizing cost"""
        # Simple cost heuristic: smaller models are cheaper
        cost_order = ["local-small", "local-medium", "local-large", "local-structured"]
        
        for model in cost_order:
            for route in routes:
                if route.get("model") == model:
                    return model
                    
        return routes[0].get("model", "local-small")
        
    def _select_by_quality(self, routes: List[Dict[str, Any]]) -> str:
        """Select model prioritizing quality"""
        # Simple quality heuristic: larger models are higher quality
        quality_order = ["local-large", "local-structured", "local-medium", "local-small"]
        
        for model in quality_order:
            for route in routes:
                if route.get("model") == model:
                    return model
                    
        return routes[0].get("model", "local-medium")
        
    def _select_by_latency(self, routes: List[Dict[str, Any]]) -> str:
        """Select model prioritizing latency"""
        # Use historical latency data if available
        best_model = None
        best_latency = float('inf')
        
        for route in routes:
            model = route.get("model")
            if model in self.usage_stats:
                avg_latency = self.usage_stats[model].get("avg_latency_ms", 1000)
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    best_model = model
                    
        return best_model or routes[0].get("model", "local-small")
        
    def _select_balanced(self, routes: List[Dict[str, Any]]) -> str:
        """Select model using balanced approach"""
        # Score models based on multiple factors
        scores = {}
        
        for route in routes:
            model = route.get("model")
            score = 0
            
            # Quality score (larger models get higher scores)
            if "large" in model:
                score += 3
            elif "medium" in model:
                score += 2
            elif "structured" in model:
                score += 2.5
            else:
                score += 1
                
            # Latency score (lower latency gets higher score)
            if model in self.usage_stats:
                avg_latency = self.usage_stats[model].get("avg_latency_ms", 1000)
                score += max(0, (2000 - avg_latency) / 1000)  # Normalize latency
                
            # Health score
            if model in self.model_health:
                health = self.model_health[model].get("success_rate", 1.0)
                score *= health
                
            scores[model] = score
            
        # Return model with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return routes[0].get("model", "local-small")
            
    async def _execute_request(self, provider: Dict[str, Any], request: LLMRequest) -> str:
        """
        Execute LLM request against provider endpoint
        """
        endpoint = provider.endpoint
        provider_type = provider.type
        
        try:
            if provider_type == "llama.cpp":
                return await self._execute_llama_cpp_request(endpoint, request)
            elif provider_type == "vllm":
                return await self._execute_vllm_request(endpoint, request)
            elif provider_type == "json-mode":
                return await self._execute_json_mode_request(endpoint, request)
            else:
                raise ValueError(f"Unsupported provider type: {provider_type}")
                
        except Exception as e:
            # Update health stats
            self._update_health_stats(request.model if hasattr(request, 'model') else 'unknown', False)
            raise e
            
    async def _execute_llama_cpp_request(self, endpoint: str, request: LLMRequest) -> str:
        """Execute request against llama.cpp server"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "prompt": request.prompt,
                "temperature": request.temperature or 0.7,
                "max_tokens": request.max_tokens or 4096,
                "stop": ["</s>", "\n\n"]
            }
            
            response = await client.post(f"{endpoint}/completion", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("content", "")
            
    async def _execute_vllm_request(self, endpoint: str, request: LLMRequest) -> str:
        """Execute request against vLLM server"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "prompt": request.prompt,
                "temperature": request.temperature or 0.3,
                "max_tokens": request.max_tokens or 8192,
                "stop": ["</s>"]
            }
            
            response = await client.post(f"{endpoint}/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("text", [""])[0]
            
    async def _execute_json_mode_request(self, endpoint: str, request: LLMRequest) -> str:
        """Execute request against JSON mode endpoint"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "prompt": request.prompt,
                "temperature": request.temperature or 0.0,
                "max_tokens": request.max_tokens or 4096,
                "response_format": {"type": "json_object"} if request.structured_output else None
            }
            
            response = await client.post(f"{endpoint}/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
            
    def _update_usage_stats(self, model: str, latency_ms: int, prompt_length: int):
        """Update usage statistics for model"""
        if model not in self.usage_stats:
            self.usage_stats[model] = {
                "total_requests": 0,
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "total_tokens": 0,
                "last_used": None
            }
            
        stats = self.usage_stats[model]
        stats["total_requests"] += 1
        stats["total_latency_ms"] += latency_ms
        stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
        stats["total_tokens"] += prompt_length
        stats["last_used"] = datetime.now().isoformat()
        
    def _update_health_stats(self, model: str, success: bool):
        """Update health statistics for model"""
        if model not in self.model_health:
            self.model_health[model] = {
                "total_requests": 0,
                "successful_requests": 0,
                "success_rate": 1.0,
                "last_failure": None
            }
            
        health = self.model_health[model]
        health["total_requests"] += 1
        
        if success:
            health["successful_requests"] += 1
        else:
            health["last_failure"] = datetime.now().isoformat()
            
        health["success_rate"] = health["successful_requests"] / health["total_requests"]
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all models"""
        return {
            "usage_stats": self.usage_stats,
            "health_stats": self.model_health,
            "total_requests": sum(stats.get("total_requests", 0) for stats in self.usage_stats.values())
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        health_results = {}
        
        for model_name, provider in self.providers.items():
            try:
                start_time = datetime.now()
                
                # Simple health check request
                test_request = LLMRequest(
                    prompt="Hello",
                    task="health_check",
                    complexity=1,
                    profile="pm",
                    max_tokens=10
                )
                
                response = await self._execute_request(provider, test_request)
                end_time = datetime.now()
                
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
                
                health_results[model_name] = {
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "endpoint": provider.endpoint,
                    "last_check": datetime.now().isoformat()
                }
                
                self._update_health_stats(model_name, True)
                
            except Exception as e:
                health_results[model_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "endpoint": provider.endpoint,
                    "last_check": datetime.now().isoformat()
                }
                
                self._update_health_stats(model_name, False)
                
        return health_results
        
    def get_recommended_model(self, task: str, complexity: int, profile: str) -> Dict[str, Any]:
        """Get model recommendation without executing request"""
        profile_config = self.profiles.get(profile, {})
        router_config = profile_config.get("llm_router", {})
        
        selected_model = self._select_model(task, complexity, router_config)
        provider = self.providers.get(selected_model)
        
        return {
            "recommended_model": selected_model,
            "endpoint": provider.endpoint if provider else None,
            "reasoning": self._get_selection_reasoning(task, complexity, profile, selected_model),
            "alternatives": self._get_alternative_models(task, complexity, router_config)
        }
        
    def _get_selection_reasoning(self, task: str, complexity: int, profile: str, selected_model: str) -> str:
        """Generate reasoning for model selection"""
        reasons = []
        
        if complexity >= 7:
            reasons.append("High complexity task requires powerful model")
        elif complexity <= 3:
            reasons.append("Low complexity task can use efficient model")
            
        if "structured" in selected_model:
            reasons.append("Structured output required")
        elif "large" in selected_model:
            reasons.append("Complex reasoning required")
        elif "small" in selected_model:
            reasons.append("Simple task, optimizing for speed")
            
        profile_config = self.profiles.get(profile, {})
        policy = profile_config.get("llm_router", {}).get("policy", "balanced")
        reasons.append(f"Using {policy} routing policy for {profile} profile")
        
        return "; ".join(reasons)
        
    def _get_alternative_models(self, task: str, complexity: int, router_config: Dict[str, Any]) -> List[str]:
        """Get alternative model suggestions"""
        all_models = list(self.providers.keys())
        selected = self._select_model(task, complexity, router_config)
        
        alternatives = [model for model in all_models if model != selected]
        
        # Sort by suitability
        if complexity >= 7:
            alternatives.sort(key=lambda x: ("large" in x, "medium" in x), reverse=True)
        else:
            alternatives.sort(key=lambda x: ("small" in x, "medium" in x), reverse=True)
            
        return alternatives[:3]  # Return top 3 alternatives
