"""
Multi-Agent Code Analysis System
Backward compatibility module - implementation in workflow/repo_workflow.py

IMPORTANT UPDATE:
The system has been upgraded to an AUTONOMOUS AGENT that uses a Supervisor LLM
to make decisions dynamically, instead of following hardcoded workflows.

This module provides backward compatibility imports for existing code.
All new code should import from workflow.repo_workflow or workflow directly.

Key Changes:
- OLD: Hardcoded workflow (Scanner -> Analyzer -> Fixer -> ...)
- NEW: Supervisor LLM decides each step based on current state
"""

# Re-export from workflow module for backward compatibility
# Note: These now use the autonomous agent system!
from workflow.repo_workflow import (
    MultiAgentRepoWorkflow as MultiAgentWorkflow,
    create_multi_agent_workflow,
    create_agent_executor
)

# Re-export tools for backward compatibility
from tools.repo_tools import ALL_REPO_TOOLS as ALL_TOOLS

# Re-export state for backward compatibility
from agent.state import RepoWorkflowState as WorkflowState

__all__ = [
    "MultiAgentWorkflow",
    "create_multi_agent_workflow",
    "create_agent_executor",
    "ALL_TOOLS",
    "WorkflowState"
]
