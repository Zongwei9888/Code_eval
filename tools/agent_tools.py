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
# AGENT-SPECIFIC TOOL SETS
# ============================================================================

# Planner Agent - 不需要工具，只做规划
PLANNER_TOOLS: List[BaseTool] = []

# Researcher Agent - 搜索和读取工具
RESEARCHER_TOOLS: List[BaseTool] = [
    grep_search,
    find_definition,
    find_references,
    get_file_symbols,
    search_files,
    read_file_with_lines,
    read_file_content,
]

# Scanner Agent - 项目扫描工具
SCANNER_TOOLS: List[BaseTool] = [
    scan_project,
    search_files,
    get_file_symbols,
]

# Analyzer Agent - 语法检查和代码分析
ANALYZER_TOOLS: List[BaseTool] = [
    check_python_syntax,
    read_file_with_lines,
    read_file_content,
    grep_search,
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

# Executor Agent - 代码执行工具
EXECUTOR_TOOLS: List[BaseTool] = [
    execute_python_file,
    run_command,
    run_background,
    kill_process,
    list_processes,
    check_port,
]

# Tester Agent - 测试执行工具
TESTER_TOOLS: List[BaseTool] = [
    run_pytest,
    run_unittest,
    run_command,
    read_file_with_lines,
]

# Reviewer Agent - 代码审查工具
REVIEWER_TOOLS: List[BaseTool] = [
    read_file_with_lines,
    read_file_content,
    grep_search,
    get_file_symbols,
    check_python_syntax,
    git_diff,
]

# Environment Agent - 环境配置工具
ENVIRONMENT_TOOLS: List[BaseTool] = [
    install_dependencies,
    run_command,
    check_port,
    get_system_info,
    read_file_content,
]

# Git Agent - 版本控制工具
GIT_AGENT_TOOLS: List[BaseTool] = [
    git_status,
    git_diff,
    git_log,
    git_add,
    git_commit,
    git_revert_file,
    git_stash,
    git_branch,
]


# ============================================================================
# AGENT TOOL REGISTRY
# ============================================================================

AGENT_TOOLS_REGISTRY: Dict[str, List[BaseTool]] = {
    "planner": PLANNER_TOOLS,
    "researcher": RESEARCHER_TOOLS,
    "scanner": SCANNER_TOOLS,
    "analyzer": ANALYZER_TOOLS,
    "fixer": FIXER_TOOLS,
    "executor": EXECUTOR_TOOLS,
    "tester": TESTER_TOOLS,
    "reviewer": REVIEWER_TOOLS,
    "environment": ENVIRONMENT_TOOLS,
    "git": GIT_AGENT_TOOLS,
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
# AGENT CAPABILITIES SUMMARY
# ============================================================================

AGENT_CAPABILITIES = {
    "planner": {
        "description": "Breaks down complex tasks into steps, creates execution plans",
        "when_to_use": "Complex tasks requiring multiple steps, task decomposition",
        "tools_count": 0,
        "key_ability": "Strategic planning and task organization"
    },
    "researcher": {
        "description": "Understands codebase context, finds relevant code",
        "when_to_use": "Need to understand dependencies, find code patterns",
        "tools_count": len(RESEARCHER_TOOLS),
        "key_ability": "Code search and context gathering"
    },
    "scanner": {
        "description": "Discovers project structure and files",
        "when_to_use": "Initial project exploration, file discovery",
        "tools_count": len(SCANNER_TOOLS),
        "key_ability": "Project structure analysis"
    },
    "analyzer": {
        "description": "Checks code for syntax errors and issues",
        "when_to_use": "Code quality check, error detection",
        "tools_count": len(ANALYZER_TOOLS),
        "key_ability": "Static analysis and error detection"
    },
    "fixer": {
        "description": "Fixes code issues with precise edits",
        "when_to_use": "Errors found, code needs modification",
        "tools_count": len(FIXER_TOOLS),
        "key_ability": "Precise code editing (str_replace)"
    },
    "executor": {
        "description": "Runs code to verify correctness",
        "when_to_use": "Need to test if code works",
        "tools_count": len(EXECUTOR_TOOLS),
        "key_ability": "Code execution and verification"
    },
    "tester": {
        "description": "Runs test suites",
        "when_to_use": "Need to run pytest/unittest",
        "tools_count": len(TESTER_TOOLS),
        "key_ability": "Test execution and reporting"
    },
    "reviewer": {
        "description": "Reviews code quality and best practices",
        "when_to_use": "Code review, quality assessment",
        "tools_count": len(REVIEWER_TOOLS),
        "key_ability": "Code quality evaluation"
    },
    "environment": {
        "description": "Manages dependencies and environment",
        "when_to_use": "Missing dependencies, environment setup",
        "tools_count": len(ENVIRONMENT_TOOLS),
        "key_ability": "Environment configuration"
    },
    "git": {
        "description": "Version control operations",
        "when_to_use": "Need to track changes, commit, or revert",
        "tools_count": len(GIT_AGENT_TOOLS),
        "key_ability": "Git operations"
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

