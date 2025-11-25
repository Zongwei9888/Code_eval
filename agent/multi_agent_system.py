"""
Multi-Agent Code Analysis System
Backward compatibility module - main implementation moved to workflow/repo_workflow.py

This module provides backward compatibility imports for existing code.
All new code should import from workflow.repo_workflow directly.
"""

# Re-export from workflow module for backward compatibility
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
