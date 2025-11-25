"""Tools module for multi-agent system"""
from .code_tools import (
    ALL_TOOLS,
    get_tool_by_name,
    read_file_tool,
    write_file_tool,
    list_files_tool,
    execute_python_code,
    execute_file,
    analyze_error,
    search_code,
    call_mcp_tool
)

# Project analyzer
from .project_analyzer import (
    ProjectAnalyzer,
    ProjectInfo,
    FileInfo,
    scan_code_bench
)

# MCP integration
try:
    from .mcp_integration import (
        MCPToolManager,
        get_mcp_tools,
        get_mcp_tools_sync,
        get_default_mcp_config
    )
    MCP_INTEGRATION_AVAILABLE = True
except ImportError:
    MCP_INTEGRATION_AVAILABLE = False
    print("[!] MCP integration not available. Install langchain-mcp-adapters to enable.")

__all__ = [
    "ALL_TOOLS",
    "get_tool_by_name",
    "read_file_tool",
    "write_file_tool",
    "list_files_tool",
    "execute_python_code",
    "execute_file",
    "analyze_error",
    "search_code",
    "call_mcp_tool"
]

if MCP_INTEGRATION_AVAILABLE:
    __all__.extend([
        "MCPToolManager",
        "get_mcp_tools",
        "get_mcp_tools_sync",
        "get_default_mcp_config"
    ])

