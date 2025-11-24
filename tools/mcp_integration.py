"""
MCP (Model Context Protocol) Integration
Real MCP tool integration using langchain-mcp-adapters
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

# Check if MCP adapters are available
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  langchain-mcp-adapters not installed. MCP integration disabled.")
    print("   Install with: pip install langchain-mcp-adapters")


class MCPToolManager:
    """
    Manager for MCP (Model Context Protocol) tools
    Handles connection to MCP servers and tool retrieval
    """
    
    def __init__(self, servers_config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize MCP tool manager
        
        Args:
            servers_config: Configuration for MCP servers
                Example:
                {
                    "filesystem": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                    },
                    "brave_search": {
                        "transport": "streamable_http",
                        "url": "http://localhost:8000/mcp"
                    }
                }
        """
        self.servers_config = servers_config or {}
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize MCP client and retrieve tools
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not MCP_AVAILABLE:
            print("⚠️  MCP not available. Skipping initialization.")
            return False
            
        if self._initialized:
            return True
            
        if not self.servers_config:
            print("ℹ️  No MCP servers configured. Skipping MCP initialization.")
            return False
            
        try:
            # Create MCP client
            self.client = MultiServerMCPClient(self.servers_config)
            
            # Retrieve tools from all configured servers
            self.tools = await self.client.get_tools()
            
            self._initialized = True
            print(f"✅ MCP initialized with {len(self.tools)} tools from {len(self.servers_config)} server(s)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize MCP: {str(e)}")
            return False
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get list of available MCP tools
        
        Returns:
            List of MCP tools (empty if not initialized)
        """
        return self.tools if self._initialized else []
    
    async def cleanup(self):
        """Clean up MCP client resources"""
        if self.client:
            try:
                await self.client.close()
                print("✅ MCP client closed")
            except Exception as e:
                print(f"⚠️  Error closing MCP client: {str(e)}")


# Default MCP configuration from environment
def get_default_mcp_config() -> Dict[str, Dict[str, Any]]:
    """
    Get default MCP configuration from environment variables
    
    Environment variables:
        MCP_ENABLED: Enable/disable MCP (default: false)
        MCP_SERVER_URL: URL for HTTP MCP server
        MCP_FILESYSTEM_PATH: Path for filesystem MCP server
        
    Returns:
        MCP servers configuration dictionary
    """
    config = {}
    
    # Check if MCP is enabled
    mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
    if not mcp_enabled:
        return config
    
    # HTTP server configuration
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    if mcp_server_url:
        config["http_server"] = {
            "transport": "streamable_http",
            "url": mcp_server_url
        }
    
    # Filesystem server configuration
    mcp_fs_path = os.getenv("MCP_FILESYSTEM_PATH")
    if mcp_fs_path:
        config["filesystem"] = {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", mcp_fs_path]
        }
    
    return config


# Global MCP manager instance
_mcp_manager: Optional[MCPToolManager] = None


async def get_mcp_tools() -> List[BaseTool]:
    """
    Get MCP tools, initializing manager if needed
    
    Returns:
        List of MCP tools
    """
    global _mcp_manager
    
    if _mcp_manager is None:
        config = get_default_mcp_config()
        _mcp_manager = MCPToolManager(config)
        await _mcp_manager.initialize()
    
    return _mcp_manager.get_tools()


def get_mcp_tools_sync() -> List[BaseTool]:
    """
    Synchronous wrapper for getting MCP tools
    
    Returns:
        List of MCP tools
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_mcp_tools())
                return future.result()
        else:
            # If no loop is running, use asyncio.run
            return asyncio.run(get_mcp_tools())
    except Exception as e:
        print(f"⚠️  Failed to get MCP tools: {str(e)}")
        return []

