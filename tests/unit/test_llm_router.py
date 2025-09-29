"""Unit tests for LLM router module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from mcp_forge.llm_router import LLMRouter, LLMProvider, RoutingPolicy, LLMError


class TestLLMProvider:
    """Test cases for LLMProvider class."""

    def test_provider_creation(self):
        """Test creating LLM provider."""
        provider = LLMProvider(
            name="test-provider",
            endpoint="http://localhost:8000",
            model_name="test-model",
            capabilities=["chat", "completion"],
            cost_per_token=0.001,
            max_tokens=4096
        )
        
        assert provider.name == "test-provider"
        assert provider.endpoint == "http://localhost:8000"
        assert provider.model_name == "test-model"
        assert provider.capabilities == ["chat", "completion"]
        assert provider.cost_per_token == 0.001
        assert provider.max_tokens == 4096
        assert provider.is_healthy is True

    def test_provider_health_tracking(self):
        """Test provider health status tracking."""
        provider = LLMProvider("test", "http://localhost:8000", "model")
        
        # Initially healthy
        assert provider.is_healthy is True
        
        # Mark as unhealthy
        provider.mark_unhealthy()
        assert provider.is_healthy is False
        
        # Mark as healthy again
        provider.mark_healthy()
        assert provider.is_healthy is True

    def test_provider_metrics_tracking(self):
        """Test provider metrics tracking."""
        provider = LLMProvider("test", "http://localhost:8000", "model")
        
        # Record request
        provider.record_request(response_time=0.5, tokens_used=100, success=True)
        
        assert provider.total_requests == 1
        assert provider.successful_requests == 1
        assert provider.total_tokens == 100
        assert provider.avg_response_time == 0.5
        
        # Record failed request
        provider.record_request(response_time=1.0, tokens_used=0, success=False)
        
        assert provider.total_requests == 2
        assert provider.successful_requests == 1
        assert provider.total_tokens == 100
        assert provider.success_rate == 0.5

    def test_provider_cost_calculation(self):
        """Test provider cost calculation."""
        provider = LLMProvider("test", "http://localhost:8000", "model", cost_per_token=0.001)
        
        cost = provider.calculate_cost(1000)
        assert cost == 1.0

    def test_provider_serialization(self):
        """Test provider serialization to dict."""
        provider = LLMProvider(
            name="test-provider",
            endpoint="http://localhost:8000",
            model_name="test-model",
            capabilities=["chat"],
            cost_per_token=0.001
        )
        
        data = provider.to_dict()
        assert data["name"] == "test-provider"
        assert data["endpoint"] == "http://localhost:8000"
        assert data["model_name"] == "test-model"
        assert data["capabilities"] == ["chat"]
        assert data["cost_per_token"] == 0.001


class TestRoutingPolicy:
    """Test cases for RoutingPolicy class."""

    def test_policy_creation(self):
        """Test creating routing policy."""
        policy = RoutingPolicy(
            profile="pm",
            difficulty_ranges={
                "small": (1, 3),
                "medium": (4, 6),
                "large": (7, 10)
            },
            provider_preferences=["small", "medium", "large"],
            fallback_strategy="round_robin"
        )
        
        assert policy.profile == "pm"
        assert policy.difficulty_ranges["small"] == (1, 3)
        assert policy.provider_preferences == ["small", "medium", "large"]
        assert policy.fallback_strategy == "round_robin"

    def test_policy_difficulty_mapping(self):
        """Test difficulty level mapping."""
        policy = RoutingPolicy(
            profile="dev",
            difficulty_ranges={
                "small": (1, 4),
                "medium": (5, 7),
                "large": (8, 10)
            }
        )
        
        assert policy.get_provider_tier(2) == "small"
        assert policy.get_provider_tier(6) == "medium"
        assert policy.get_provider_tier(9) == "large"
        assert policy.get_provider_tier(11) == "large"  # Above max should use largest

    def test_policy_serialization(self):
        """Test policy serialization."""
        policy = RoutingPolicy(
            profile="tpm",
            difficulty_ranges={"small": (1, 5), "large": (6, 10)},
            provider_preferences=["small", "large"]
        )
        
        data = policy.to_dict()
        assert data["profile"] == "tpm"
        assert data["difficulty_ranges"] == {"small": (1, 5), "large": (6, 10)}
        assert data["provider_preferences"] == ["small", "large"]


class TestLLMRouter:
    """Test cases for LLMRouter class."""

    @pytest.fixture
    def sample_providers(self):
        """Create sample LLM providers for testing."""
        return {
            "small": LLMProvider(
                name="small-model",
                endpoint="http://localhost:9001",
                model_name="llama-7b",
                capabilities=["chat", "completion"],
                cost_per_token=0.0001,
                max_tokens=2048
            ),
            "medium": LLMProvider(
                name="medium-model",
                endpoint="http://localhost:9002",
                model_name="llama-13b",
                capabilities=["chat", "completion", "structured"],
                cost_per_token=0.0005,
                max_tokens=4096
            ),
            "large": LLMProvider(
                name="large-model",
                endpoint="http://localhost:9003",
                model_name="llama-70b",
                capabilities=["chat", "completion", "structured", "reasoning"],
                cost_per_token=0.002,
                max_tokens=8192
            )
        }

    @pytest.fixture
    def sample_policies(self):
        """Create sample routing policies."""
        return {
            "pm": RoutingPolicy(
                profile="pm",
                difficulty_ranges={
                    "small": (1, 4),
                    "medium": (5, 7),
                    "large": (8, 10)
                },
                provider_preferences=["small", "medium", "large"],
                fallback_strategy="cost_optimized"
            ),
            "tpm": RoutingPolicy(
                profile="tpm",
                difficulty_ranges={
                    "small": (1, 3),
                    "medium": (4, 6),
                    "large": (7, 10)
                },
                provider_preferences=["medium", "large", "small"],
                fallback_strategy="quality_first"
            ),
            "dev": RoutingPolicy(
                profile="dev",
                difficulty_ranges={
                    "small": (1, 5),
                    "medium": (6, 8),
                    "large": (9, 10)
                },
                provider_preferences=["small", "medium", "large"],
                fallback_strategy="round_robin"
            )
        }

    @pytest.fixture
    def router(self, sample_providers, sample_policies):
        """Create LLMRouter instance with sample data."""
        return LLMRouter(providers=sample_providers, policies=sample_policies)

    def test_router_initialization(self, router, sample_providers, sample_policies):
        """Test router initialization."""
        assert len(router.providers) == 3
        assert len(router.policies) == 3
        assert "small" in router.providers
        assert "pm" in router.policies

    def test_select_provider_by_difficulty(self, router):
        """Test provider selection based on difficulty."""
        # Low difficulty should select small provider
        provider = router.select_provider(profile="pm", difficulty=2)
        assert provider.name == "small-model"
        
        # Medium difficulty should select medium provider
        provider = router.select_provider(profile="pm", difficulty=6)
        assert provider.name == "medium-model"
        
        # High difficulty should select large provider
        provider = router.select_provider(profile="pm", difficulty=9)
        assert provider.name == "large-model"

    def test_select_provider_with_capability_requirement(self, router):
        """Test provider selection with capability requirements."""
        # Request structured output capability
        provider = router.select_provider(
            profile="pm", 
            difficulty=3, 
            required_capabilities=["structured"]
        )
        # Should select medium or large (both have structured capability)
        assert provider.name in ["medium-model", "large-model"]

    def test_select_provider_fallback_unhealthy(self, router):
        """Test provider selection fallback when preferred provider is unhealthy."""
        # Mark small provider as unhealthy
        router.providers["small"].mark_unhealthy()
        
        # Request low difficulty (would normally use small)
        provider = router.select_provider(profile="pm", difficulty=2)
        # Should fallback to next available healthy provider
        assert provider.name != "small-model"
        assert provider.is_healthy

    def test_select_provider_cost_optimized_fallback(self, router):
        """Test cost-optimized fallback strategy."""
        # Mark preferred provider unhealthy to trigger fallback
        router.providers["small"].mark_unhealthy()
        
        provider = router.select_provider(profile="pm", difficulty=2)
        # Should select the cheapest available healthy provider
        assert provider.is_healthy

    def test_select_provider_quality_first_fallback(self, router):
        """Test quality-first fallback strategy."""
        # Mark preferred provider unhealthy
        router.providers["medium"].mark_unhealthy()
        
        provider = router.select_provider(profile="tpm", difficulty=5)
        # Should select highest quality available provider
        assert provider.is_healthy

    def test_select_provider_round_robin_fallback(self, router):
        """Test round-robin fallback strategy."""
        # Mark preferred provider unhealthy
        router.providers["small"].mark_unhealthy()
        
        # Make multiple requests to test round-robin
        providers = []
        for _ in range(4):
            provider = router.select_provider(profile="dev", difficulty=3)
            providers.append(provider.name)
        
        # Should cycle through available providers
        assert len(set(providers)) > 1

    def test_select_provider_no_healthy_providers(self, router):
        """Test provider selection when no providers are healthy."""
        # Mark all providers as unhealthy
        for provider in router.providers.values():
            provider.mark_unhealthy()
        
        with pytest.raises(LLMError, match="No healthy providers available"):
            router.select_provider(profile="pm", difficulty=5)

    def test_select_provider_unknown_profile(self, router):
        """Test provider selection with unknown profile."""
        with pytest.raises(LLMError, match="Unknown profile"):
            router.select_provider(profile="unknown", difficulty=5)

    @pytest.mark.asyncio
    async def test_route_request_success(self, router):
        """Test successful request routing."""
        with patch.object(router, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "response": "Test response",
                "tokens_used": 100,
                "response_time": 0.5
            }
            
            result = await router.route_request(
                profile="pm",
                difficulty=5,
                prompt="Test prompt",
                task_type="requirements"
            )
            
            assert result["response"] == "Test response"
            assert result["tokens_used"] == 100
            assert result["provider_used"] is not None
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_request_with_retry(self, router):
        """Test request routing with retry on failure."""
        with patch.object(router, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            # First call fails, second succeeds
            mock_request.side_effect = [
                Exception("Network error"),
                {
                    "response": "Test response",
                    "tokens_used": 100,
                    "response_time": 0.5
                }
            ]
            
            result = await router.route_request(
                profile="pm",
                difficulty=5,
                prompt="Test prompt",
                max_retries=2
            )
            
            assert result["response"] == "Test response"
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_route_request_max_retries_exceeded(self, router):
        """Test request routing when max retries are exceeded."""
        with patch.object(router, '_make_llm_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Persistent error")
            
            with pytest.raises(LLMError, match="Max retries exceeded"):
                await router.route_request(
                    profile="pm",
                    difficulty=5,
                    prompt="Test prompt",
                    max_retries=2
                )

    def test_health_check_all_healthy(self, router):
        """Test health check when all providers are healthy."""
        health = router.health_check()
        
        assert health["overall_status"] == "healthy"
        assert len(health["providers"]) == 3
        assert all(p["is_healthy"] for p in health["providers"].values())

    def test_health_check_some_unhealthy(self, router):
        """Test health check when some providers are unhealthy."""
        router.providers["small"].mark_unhealthy()
        
        health = router.health_check()
        
        assert health["overall_status"] == "degraded"
        assert not health["providers"]["small"]["is_healthy"]
        assert health["providers"]["medium"]["is_healthy"]

    def test_health_check_all_unhealthy(self, router):
        """Test health check when all providers are unhealthy."""
        for provider in router.providers.values():
            provider.mark_unhealthy()
        
        health = router.health_check()
        
        assert health["overall_status"] == "unhealthy"
        assert not any(p["is_healthy"] for p in health["providers"].values())

    def test_get_metrics(self, router):
        """Test getting router metrics."""
        # Record some metrics
        router.providers["small"].record_request(0.5, 100, True)
        router.providers["medium"].record_request(1.0, 200, False)
        
        metrics = router.get_metrics()
        
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "total_tokens" in metrics
        assert "providers" in metrics
        assert len(metrics["providers"]) == 3

    def test_add_provider(self, router):
        """Test adding new provider."""
        new_provider = LLMProvider(
            name="new-provider",
            endpoint="http://localhost:9004",
            model_name="new-model"
        )
        
        router.add_provider("new", new_provider)
        
        assert "new" in router.providers
        assert router.providers["new"] == new_provider

    def test_remove_provider(self, router):
        """Test removing provider."""
        router.remove_provider("small")
        
        assert "small" not in router.providers
        assert len(router.providers) == 2

    def test_update_policy(self, router):
        """Test updating routing policy."""
        new_policy = RoutingPolicy(
            profile="pm",
            difficulty_ranges={"small": (1, 5), "large": (6, 10)},
            provider_preferences=["large", "small"]
        )
        
        router.update_policy("pm", new_policy)
        
        assert router.policies["pm"] == new_policy

    @patch('httpx.AsyncClient.post')
    @pytest.mark.asyncio
    async def test_make_llm_request_success(self, mock_post, router):
        """Test successful LLM request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 100}
        }
        mock_post.return_value = mock_response
        
        provider = router.providers["small"]
        result = await router._make_llm_request(provider, "Test prompt")
        
        assert result["response"] == "Test response"
        assert result["tokens_used"] == 100
        assert result["response_time"] > 0

    @patch('httpx.AsyncClient.post')
    @pytest.mark.asyncio
    async def test_make_llm_request_http_error(self, mock_post, router):
        """Test LLM request with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        provider = router.providers["small"]
        
        with pytest.raises(LLMError, match="HTTP 500"):
            await router._make_llm_request(provider, "Test prompt")

    def test_estimate_cost(self, router):
        """Test cost estimation for request."""
        cost = router.estimate_cost(profile="pm", difficulty=5, estimated_tokens=1000)
        
        # Should return cost based on selected provider
        assert isinstance(cost, float)
        assert cost > 0

    def test_get_provider_recommendations(self, router):
        """Test getting provider recommendations."""
        recommendations = router.get_provider_recommendations(
            profile="pm",
            difficulty=6,
            required_capabilities=["structured"]
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all("provider" in rec for rec in recommendations)
        assert all("score" in rec for rec in recommendations)

    def test_router_serialization(self, router):
        """Test router serialization to dict."""
        data = router.to_dict()
        
        assert "providers" in data
        assert "policies" in data
        assert "metrics" in data
        assert len(data["providers"]) == 3
        assert len(data["policies"]) == 3


class TestLLMError:
    """Test LLMError exception class."""

    def test_error_creation(self):
        """Test creating LLMError."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_error_with_provider_info(self):
        """Test creating LLMError with provider information."""
        error = LLMError("Request failed", provider="test-provider")
        assert error.provider == "test-provider"
        assert "Request failed" in str(error)


if __name__ == "__main__":
    pytest.main([__file__])
