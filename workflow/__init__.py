"""
Workflow Module
LangGraph-based workflows for code analysis and improvement
"""

# Single-file workflow
from .code_workflow import (
    CodeImprovementWorkflow,
    create_workflow
)

# Repository multi-agent workflow
from .repo_workflow import (
    MultiAgentRepoWorkflow,
    create_multi_agent_workflow,
    create_agent_executor
)

# Legacy interactive session
try:
    from .code_workflow import interactive_session
except ImportError:
    interactive_session = None

__all__ = [
    # Single-file workflow
    "CodeImprovementWorkflow",
    "create_workflow",
    # Repository workflow
    "MultiAgentRepoWorkflow",
    "create_multi_agent_workflow",
    "create_agent_executor",
    # Legacy
    "interactive_session"
]
