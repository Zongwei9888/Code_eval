"""Agent module for multi-agent system"""
from .code_agents import (
    CodeAnalyzerAgent,
    CodeExecutorAgent,
    CodeModifierAgent,
    create_agents
)
from .state import MultiAgentState, create_initial_state

# LangGraph-based repo workflow (proper multi-agent with LLM + tools)
from .repo_workflow import (
    RepoAnalysisWorkflow,
    RepoAnalysisState,
    create_repo_workflow,
    ALL_REPO_TOOLS,
    # Tools
    scan_directory,
    read_file,
    write_file,
    check_syntax,
    run_python_file,
    run_tests,
    install_dependencies
)

# Legacy repo agents (local utilities)
from .repo_agents import (
    ProjectScannerAgent,
    StaticAnalyzerAgent,
    EnvironmentAgent,
    TestRunnerAgent,
    ErrorAnalyzerAgent,
    CodeFixerAgent,
    RepoAnalysisOrchestrator,
    create_orchestrator,
    quick_scan,
    ProjectInfo,
    FileAnalysis,
    AnalysisReport
)

__all__ = [
    # Original agents
    "CodeAnalyzerAgent",
    "CodeExecutorAgent",
    "CodeModifierAgent",
    "MultiAgentState",
    "create_agents",
    "create_initial_state",
    # New LangGraph workflow
    "RepoAnalysisWorkflow",
    "RepoAnalysisState",
    "create_repo_workflow",
    "ALL_REPO_TOOLS",
    # Legacy repo utilities
    "ProjectScannerAgent",
    "StaticAnalyzerAgent",
    "EnvironmentAgent",
    "TestRunnerAgent",
    "ErrorAnalyzerAgent",
    "CodeFixerAgent",
    "RepoAnalysisOrchestrator",
    "create_orchestrator",
    "quick_scan",
    "ProjectInfo",
    "FileAnalysis",
    "AnalysisReport"
]

