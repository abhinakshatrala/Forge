"""
Main application entry point for MCP Forge
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .config import MCPConfig
from .server import MCPServer
from .ui import BusinessUserUI
from .workflow_engine import WorkflowEngine


logger = logging.getLogger(__name__)


class MCPForgeApp:
    """
    Main MCP Forge application
    Coordinates all components and handles graceful shutdown
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig.load_from_env()
        self.config.ensure_directories()
        
        # Initialize components
        self.mcp_server = MCPServer(self.config)
        self.ui_server = BusinessUserUI(self.config)
        self.workflow_engine = WorkflowEngine(self.config)
        
        # Track running tasks
        self.tasks = []
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def start_all_services(self):
        """Start all services concurrently"""
        logger.info("Starting MCP Forge application...")
        
        try:
            # Start MCP server on stdio
            mcp_task = asyncio.create_task(
                self.mcp_server.run_stdio(),
                name="mcp_server"
            )
            self.tasks.append(mcp_task)
            
            # Start UI server if enabled
            if self.config.ui_enabled:
                import uvicorn
                ui_config = uvicorn.Config(
                    app=self.ui_server.get_app(),
                    host=self.config.ui_bind.split(':')[0],
                    port=int(self.config.ui_bind.split(':')[1]),
                    log_level="info"
                )
                ui_server = uvicorn.Server(ui_config)
                ui_task = asyncio.create_task(
                    ui_server.serve(),
                    name="ui_server"
                )
                self.tasks.append(ui_task)
                
            # Start background workflow processor
            workflow_task = asyncio.create_task(
                self._run_workflow_processor(),
                name="workflow_processor"
            )
            self.tasks.append(workflow_task)
            
            # Start health monitor
            health_task = asyncio.create_task(
                self._run_health_monitor(),
                name="health_monitor"
            )
            self.tasks.append(health_task)
            
            logger.info("All services started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            raise
        finally:
            await self._shutdown()
            
    async def _run_workflow_processor(self):
        """Background workflow processor"""
        logger.info("Starting workflow processor...")
        
        try:
            while not self.shutdown_event.is_set():
                # Process any pending workflows
                pending_workflows = self.workflow_engine.list_workflows(status_filter="pending")
                
                for workflow_info in pending_workflows:
                    try:
                        logger.info(f"Processing workflow: {workflow_info['id']}")
                        await self.workflow_engine.execute_workflow(workflow_info['id'])
                    except Exception as e:
                        logger.error(f"Error processing workflow {workflow_info['id']}: {e}")
                        
                # Wait before next check
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            logger.info("Workflow processor cancelled")
        except Exception as e:
            logger.error(f"Workflow processor error: {e}")
            
    async def _run_health_monitor(self):
        """Background health monitor"""
        logger.info("Starting health monitor...")
        
        try:
            while not self.shutdown_event.is_set():
                # Check LLM provider health
                from .llm_router import LLMRouter
                router = LLMRouter(self.config)
                
                try:
                    health_results = await router.health_check()
                    unhealthy_providers = [
                        name for name, result in health_results.items()
                        if result['status'] != 'healthy'
                    ]
                    
                    if unhealthy_providers:
                        logger.warning(f"Unhealthy LLM providers: {unhealthy_providers}")
                    else:
                        logger.debug("All LLM providers healthy")
                        
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.info("Health monitor cancelled")
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            
    async def _shutdown(self):
        """Graceful shutdown of all services"""
        logger.info("Shutting down MCP Forge application...")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        logger.info("Shutdown complete")
        
    async def run_mcp_server_only(self, transport: str = "stdio", **kwargs):
        """Run only the MCP server"""
        logger.info(f"Starting MCP server with {transport} transport...")
        
        try:
            if transport == "stdio":
                await self.mcp_server.run_stdio()
            elif transport == "unix":
                socket_path = kwargs.get("socket_path", "/run/mcp.sock")
                await self.mcp_server.run_unix_socket(socket_path)
            elif transport == "https":
                host = kwargs.get("host", "127.0.0.1")
                port = kwargs.get("port", 8443)
                await self.mcp_server.run_https(host, port)
            else:
                raise ValueError(f"Unknown transport: {transport}")
                
        except KeyboardInterrupt:
            logger.info("MCP server stopped by user")
        except Exception as e:
            logger.error(f"MCP server error: {e}")
            raise
            
    async def run_ui_only(self, host: str = "127.0.0.1", port: int = 8788):
        """Run only the UI server"""
        logger.info(f"Starting UI server on {host}:{port}...")
        
        try:
            import uvicorn
            config = uvicorn.Config(
                app=self.ui_server.get_app(),
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except KeyboardInterrupt:
            logger.info("UI server stopped by user")
        except Exception as e:
            logger.error(f"UI server error: {e}")
            raise


def setup_logging(log_level: str = "INFO"):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mcp-forge.log')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    """Main application entry point"""
    setup_logging()
    
    try:
        app = MCPForgeApp()
        await app.start_all_services()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
