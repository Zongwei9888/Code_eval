"""Workflow module for multi-agent system"""
from .code_workflow import (
    interactive_session
)

# Import improved workflow with memory support
try:
    from .code_workflow_improved import (
        CodeImprovementWorkflow,
        create_workflow
    )
    print("✅ Using improved workflow with memory/checkpointing support")
except ImportError as e:
    print(f"⚠️  Could not import improved workflow: {e}")
    print("   Falling back to basic workflow")
    from .code_workflow import (
        CodeImprovementWorkflow,
        create_workflow
    )

__all__ = [
    "CodeImprovementWorkflow",
    "create_workflow",
    "interactive_session"
]

