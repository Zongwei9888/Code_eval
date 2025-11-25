"""
State Definitions for Multi-Agent Code Analysis System
Consolidated state definitions following LangGraph best practices
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# ============================================================================
# SINGLE FILE WORKFLOW STATE
# ============================================================================

class SingleFileState(TypedDict):
    """
    State for single-file code improvement workflow.
    Uses TypedDict for proper LangGraph integration.
    """
    # Message history with proper annotation
    messages: Annotated[List[AnyMessage], add_messages]
    
    # File information
    target_file: str
    file_content: str
    
    # Analysis results
    analysis_complete: bool
    code_analysis: str
    identified_issues: List[str]
    
    # Execution results
    execution_attempts: int
    last_execution_result: str
    execution_success: bool
    last_error: str
    
    # Modification tracking
    modification_history: List[Dict[str, Any]]
    current_code: str
    
    # Workflow control
    max_attempts: int
    should_continue: bool
    final_status: str


# Alias for backward compatibility
MultiAgentState = SingleFileState


# ============================================================================
# REPO WORKFLOW STATE
# ============================================================================

class RepoWorkflowState(TypedDict):
    """
    State for multi-agent repository analysis workflow.
    Supports the full analysis pipeline: scan -> analyze -> fix -> execute -> report
    """
    # Core message history (this IS the memory)
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Project info
    project_path: str
    project_name: str
    
    # Files discovered
    python_files: List[str]
    test_files: List[str]
    
    # Analysis results
    files_with_errors: List[Dict[str, Any]]
    current_file: Optional[str]
    
    # Execution tracking
    execution_results: List[Dict[str, Any]]
    last_execution_success: bool
    
    # Fix tracking
    fix_attempts: int
    max_fix_attempts: int
    fixes_applied: List[Dict[str, Any]]
    
    # Workflow control
    workflow_complete: bool
    final_status: str
    
    # Logging for UI
    step_logs: List[Dict[str, Any]]


# ============================================================================
# AGENT STATE (for individual agent loops)
# ============================================================================

class AgentLoopState(TypedDict):
    """
    State for a single agent with tool calling loop.
    Following LangGraph's MessagesState pattern.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    agent_logs: List[Dict[str, Any]]


# ============================================================================
# STATE FACTORY FUNCTIONS
# ============================================================================

def create_single_file_state(
    target_file: str,
    initial_code: str = "",
    max_attempts: int = 5
) -> SingleFileState:
    """
    Create initial state for single-file workflow.
    
    Args:
        target_file: Path to the target file
        initial_code: Initial code content
        max_attempts: Maximum execution attempts
        
    Returns:
        Initial state dictionary
    """
    return {
        "messages": [],
        "target_file": target_file,
        "file_content": initial_code,
        "analysis_complete": False,
        "code_analysis": "",
        "identified_issues": [],
        "execution_attempts": 0,
        "last_execution_result": "",
        "execution_success": False,
        "last_error": "",
        "modification_history": [],
        "current_code": initial_code,
        "max_attempts": max_attempts,
        "should_continue": True,
        "final_status": ""
    }


# Alias for backward compatibility
create_initial_state = create_single_file_state


def create_repo_workflow_state(
    project_path: str,
    max_fix_attempts: int = 5
) -> RepoWorkflowState:
    """
    Create initial state for repository workflow.
    
    Args:
        project_path: Path to the project directory
        max_fix_attempts: Maximum fix attempts per file
        
    Returns:
        Initial state dictionary
    """
    from pathlib import Path
    
    return {
        "messages": [],
        "project_path": str(Path(project_path).resolve()),
        "project_name": Path(project_path).name,
        "python_files": [],
        "test_files": [],
        "files_with_errors": [],
        "current_file": None,
        "execution_results": [],
        "last_execution_success": False,
        "fix_attempts": 0,
        "max_fix_attempts": max_fix_attempts,
        "fixes_applied": [],
        "workflow_complete": False,
        "final_status": "",
        "step_logs": []
    }


def create_agent_loop_state() -> AgentLoopState:
    """
    Create initial state for agent loop.
    
    Returns:
        Initial agent loop state
    """
    return {
        "messages": [],
        "agent_logs": []
    }
