"""
Agent-Specific Tool Mappings

每个代理都有专门的工具集，这样可以：
1. 限制代理只能使用适合其职责的工具
2. 减少LLM的困惑
3. 提高任务专注度

工具映射原则：
- Planner: 无工具，只做规划
- Researcher: 搜索和读取工具
- Scanner: 项目扫描工具
- Analyzer: 语法检查和代码分析工具
- Fixer: 精确编辑工具
- Executor: 执行和命令工具
- Tester: 测试运行工具
- Reviewer: 代码审查工具
"""

from typing import List, Dict, Any
from langchain_core.tools import BaseTool

# Import all tools
from .advanced_edit_tools import (
    str_replace,
    insert_at_line,
    delete_lines,
    apply_diff,
    read_file_with_lines,
    undo_edit,
    get_edit_history,
    ADVANCED_EDIT_TOOLS
)

from .search_tools import (
    grep_search,
    find_definition,
    find_references,
    get_file_symbols,
    search_files,
    SEARCH_TOOLS
)

from .git_tools import (
    git_status,
    git_diff,
    git_log,
    git_show,
    git_add,
    git_commit,
    git_revert_file,
    git_stash,
    git_branch,
    GIT_TOOLS
)

from .terminal_tools import (
    run_command,
    run_background,
    kill_process,
    list_processes,
    get_process_output,
    check_port,
    get_system_info,
    TERMINAL_TOOLS
)

from .repo_tools import (
    scan_project,
    read_file_content,
    write_file_content,
    check_python_syntax,
    execute_python_file,
    run_pytest,
    run_unittest,
    install_dependencies,
    ALL_REPO_TOOLS
)


# ============================================================================
# CORE AGENT TOOL SETS (4 Core Agents)
# ============================================================================

# Scanner Agent - 项目扫描和发现
SCANNER_TOOLS: List[BaseTool] = [
    scan_project,
    search_files,
    get_file_symbols,
    read_file_content,
]

# Analyzer Agent - 代码分析（合并了researcher功能）
ANALYZER_TOOLS: List[BaseTool] = [
    check_python_syntax,
    read_file_with_lines,
    read_file_content,
    grep_search,
    find_definition,
    find_references,
    get_file_symbols,
]

# Fixer Agent - 精确编辑工具 (核心!)
FIXER_TOOLS: List[BaseTool] = [
    str_replace,          # 精确替换 - 主要工具
    insert_at_line,       # 行插入
    delete_lines,         # 行删除
    apply_diff,           # 应用diff
    read_file_with_lines, # 带行号读取
    undo_edit,            # 撤销
    grep_search,          # 搜索上下文
]

# Executor Agent - 执行和环境管理（精简版）
EXECUTOR_TOOLS: List[BaseTool] = [
    run_command,           # 核心：执行任何命令（python, npm, pytest等）
    install_dependencies,  # 环境：安装依赖
    read_file_content,     # 辅助：读取配置文件、检查入口点
]


# ============================================================================
# AGENT TOOL REGISTRY (Core 4 Agents)
# ============================================================================

AGENT_TOOLS_REGISTRY: Dict[str, List[BaseTool]] = {
    "scanner": SCANNER_TOOLS,
    "analyzer": ANALYZER_TOOLS,
    "fixer": FIXER_TOOLS,
    "executor": EXECUTOR_TOOLS,
}


def get_tools_for_agent(agent_name: str) -> List[BaseTool]:
    """
    获取指定代理的工具集
    
    Args:
        agent_name: 代理名称
        
    Returns:
        该代理专用的工具列表
    """
    return AGENT_TOOLS_REGISTRY.get(agent_name.lower(), [])


def get_all_tools() -> List[BaseTool]:
    """获取所有可用工具（去重）"""
    all_tools = []
    seen_names = set()
    
    for tools in AGENT_TOOLS_REGISTRY.values():
        for tool in tools:
            if tool.name not in seen_names:
                all_tools.append(tool)
                seen_names.add(tool.name)
    
    return all_tools


def get_tool_descriptions() -> str:
    """获取所有工具的描述文档"""
    lines = ["# Available Tools by Agent\n"]
    
    for agent_name, tools in AGENT_TOOLS_REGISTRY.items():
        lines.append(f"\n## {agent_name.upper()} Agent\n")
        if not tools:
            lines.append("- (No tools - planning/reasoning only)\n")
        else:
            for tool in tools:
                lines.append(f"- **{tool.name}**: {tool.description[:100]}...")
    
    return "\n".join(lines)


# ============================================================================
# AGENT CAPABILITIES SUMMARY (4 Core Agents)
# ============================================================================

AGENT_CAPABILITIES = {
    "scanner": {
        "description": "Discovers project structure and files",
        "when_to_use": "Initial project exploration, file discovery",
        "tools_count": len(SCANNER_TOOLS),
        "key_ability": "Project structure analysis"
    },
    "analyzer": {
        "description": "Analyzes code, checks syntax, searches patterns",
        "when_to_use": "Code analysis, syntax check, find definitions",
        "tools_count": len(ANALYZER_TOOLS),
        "key_ability": "Static analysis and code understanding"
    },
    "fixer": {
        "description": "Fixes code issues with precise edits",
        "when_to_use": "Errors found, code needs modification",
        "tools_count": len(FIXER_TOOLS),
        "key_ability": "Precise code editing (str_replace)"
    },
    "executor": {
        "description": "Runs code, tests, and manages dependencies",
        "when_to_use": "Execute code, run tests, install packages",
        "tools_count": len(EXECUTOR_TOOLS),
        "key_ability": "Code execution, testing, environment management"
    },
}


def get_agent_summary() -> str:
    """获取所有代理的能力摘要"""
    lines = ["# Agent Capabilities Summary\n"]
    
    for name, info in AGENT_CAPABILITIES.items():
        lines.append(f"\n## {name.upper()}")
        lines.append(f"- **Description**: {info['description']}")
        lines.append(f"- **When to use**: {info['when_to_use']}")
        lines.append(f"- **Tools**: {info['tools_count']}")
        lines.append(f"- **Key ability**: {info['key_ability']}")
    
    return "\n".join(lines)

