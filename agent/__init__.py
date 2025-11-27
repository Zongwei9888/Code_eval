"""
Agent Module
Multi-agent system components for code analysis and improvement
"""

# State definitions
from .state import (
    SingleFileState,
    MultiAgentState,  # Alias for backward compatibility
    RepoWorkflowState,
    AgentLoopState,
    TrueAgentState,  # New: for autonomous agent
    create_single_file_state,
    create_initial_state,  # Alias for backward compatibility
    create_repo_workflow_state,
    create_agent_loop_state,
    create_true_agent_state  # New: for autonomous agent
)

# Single-file agents
from .code_agents import (
    CodeAnalyzerAgent,
    CodeExecutorAgent,
    CodeModifierAgent,
    create_agents
)

# Local repository agents (non-LLM)
from .repo_agents import (
    ProjectScannerAgent,
    StaticAnalyzerAgent,
    EnvironmentAgent,
    TestRunnerAgent,
    RepoAnalysisOrchestrator,
    create_orchestrator,
    quick_scan,
    # Data classes
    ProjectInfo,
    FileAnalysis,
    TestResult,
    AnalysisReport
)

# Multi-agent workflow (backward compatibility imports)
from .multi_agent_system import (
    MultiAgentWorkflow,
    create_multi_agent_workflow,
    ALL_TOOLS
)

__all__ = [
    # State
    "SingleFileState",
    "MultiAgentState",
    "RepoWorkflowState",
    "AgentLoopState",
    "TrueAgentState",
    "create_single_file_state",
    "create_initial_state",
    "create_repo_workflow_state",
    "create_agent_loop_state",
    "create_true_agent_state",
    # Single-file agents
    "CodeAnalyzerAgent",
    "CodeExecutorAgent",
    "CodeModifierAgent",
    "create_agents",
    # Local repo agents
    "ProjectScannerAgent",
    "StaticAnalyzerAgent",
    "EnvironmentAgent",
    "TestRunnerAgent",
    "RepoAnalysisOrchestrator",
    "create_orchestrator",
    "quick_scan",
    "ProjectInfo",
    "FileAnalysis",
    "TestResult",
    "AnalysisReport",
    # Multi-agent workflow
    "MultiAgentWorkflow",
    "create_multi_agent_workflow",
    "ALL_TOOLS"
]
