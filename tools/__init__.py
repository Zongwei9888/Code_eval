"""
Tools Module
Code manipulation and analysis tools for multi-agent system
"""

# Single-file tools
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

# Repository analysis tools
from .repo_tools import (
    ALL_REPO_TOOLS,
    SCANNER_TOOLS,
    ANALYZER_TOOLS,
    FIXER_TOOLS,
    EXECUTOR_TOOLS,
    get_repo_tool_by_name,
    scan_project,
    read_file_content,
    write_file_content,
    check_python_syntax,
    execute_python_file,
    run_pytest,
    run_unittest,
    install_dependencies
)

# Project analyzer utility
from .project_analyzer import (
    ProjectAnalyzer,
    ProjectInfo,
    FileInfo,
    scan_code_bench
)

# MCP integration (optional)
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

__all__ = [
    # Single-file tools
    "ALL_TOOLS",
    "get_tool_by_name",
    "read_file_tool",
    "write_file_tool",
    "list_files_tool",
    "execute_python_code",
    "execute_file",
    "analyze_error",
    "search_code",
    "call_mcp_tool",
    # Repository tools
    "ALL_REPO_TOOLS",
    "SCANNER_TOOLS",
    "ANALYZER_TOOLS",
    "FIXER_TOOLS",
    "EXECUTOR_TOOLS",
    "get_repo_tool_by_name",
    "scan_project",
    "read_file_content",
    "write_file_content",
    "check_python_syntax",
    "execute_python_file",
    "run_pytest",
    "run_unittest",
    "install_dependencies",
    # Project analyzer
    "ProjectAnalyzer",
    "ProjectInfo",
    "FileInfo",
    "scan_code_bench",
    # MCP
    "MCP_INTEGRATION_AVAILABLE"
]

if MCP_INTEGRATION_AVAILABLE:
    __all__.extend([
        "MCPToolManager",
        "get_mcp_tools",
        "get_mcp_tools_sync",
        "get_default_mcp_config"
    ])
