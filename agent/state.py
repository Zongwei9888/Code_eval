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
    
    Updated to support autonomous agent decision-making.
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
    
    # Supervisor decision (for autonomous agent mode)
    supervisor_decision: str
    current_task: str


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
        "final_status": "",
        # Supervisor decision fields (for autonomous mode)
        "supervisor_decision": "",
        "current_task": ""
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


# ============================================================================
# TRUE AGENT STATE (Autonomous Decision-Making Agent)
# ============================================================================

class TrueAgentState(TypedDict):
    """
    Shared State for True Autonomous Agent System.
    
    This state is shared across all specialist agents and the supervisor.
    The supervisor observes this state and decides which agent to invoke next.
    
    Key Design Principles:
    1. All context is shared - every agent can see the full picture
    2. Supervisor makes decisions based on this state
    3. Supports dynamic routing and feedback loops
    """
    # ==========================================================================
    # Core Message History (Memory)
    # ==========================================================================
    messages: Annotated[List[AnyMessage], add_messages]
    
    # ==========================================================================
    # User Request & Goal
    # ==========================================================================
    user_request: str  # Original user request
    goal_achieved: bool  # Has the goal been achieved?
    
    # ==========================================================================
    # Project Context (Shared Knowledge)
    # ==========================================================================
    project_path: str
    project_name: str
    python_files: List[str]  # All discovered Python files
    test_files: List[str]  # All discovered test files
    
    # ==========================================================================
    # Current Focus
    # ==========================================================================
    current_file: Optional[str]  # File currently being worked on
    current_task: str  # Current task description
    
    # ==========================================================================
    # Analysis Results (Shared Knowledge)
    # ==========================================================================
    syntax_errors: List[Dict[str, Any]]  # Files with syntax errors
    runtime_errors: List[Dict[str, Any]]  # Runtime errors encountered
    test_failures: List[Dict[str, Any]]  # Failed tests
    
    # ==========================================================================
    # Execution History (Feedback Loop)
    # ==========================================================================
    execution_history: List[Dict[str, Any]]  # All execution attempts
    last_execution_success: bool
    last_error_message: str
    
    # ==========================================================================
    # Modification History (Audit Trail)
    # ==========================================================================
    modifications: List[Dict[str, Any]]  # All code changes made
    
    # ==========================================================================
    # Supervisor Decision Tracking
    # ==========================================================================
    supervisor_decision: str  # Current decision (which agent to call)
    supervisor_reasoning: str  # Why this decision was made
    decision_history: List[Dict[str, Any]]  # All decisions made
    
    # ==========================================================================
    # Loop Control
    # ==========================================================================
    iteration_count: int  # How many supervisor iterations
    max_iterations: int  # Maximum allowed iterations
    
    # ==========================================================================
    # Final Output
    # ==========================================================================
    final_report: str
    step_logs: List[Dict[str, Any]]  # For UI display


def create_true_agent_state(
    project_path: str,
    user_request: str = "Analyze and fix code issues",
    max_iterations: int = 20
) -> TrueAgentState:
    """
    Create initial state for True Agent System.
    
    Args:
        project_path: Path to the project directory
        user_request: User's request/goal
        max_iterations: Maximum supervisor iterations to prevent infinite loops
        
    Returns:
        Initial TrueAgentState dictionary
    """
    from pathlib import Path
    
    return {
        # Messages
        "messages": [],
        
        # User request
        "user_request": user_request,
        "goal_achieved": False,
        
        # Project context
        "project_path": str(Path(project_path).resolve()),
        "project_name": Path(project_path).name,
        "python_files": [],
        "test_files": [],
        
        # Current focus
        "current_file": None,
        "current_task": "",
        
        # Analysis results
        "syntax_errors": [],
        "runtime_errors": [],
        "test_failures": [],
        
        # Execution history
        "execution_history": [],
        "last_execution_success": False,
        "last_error_message": "",
        
        # Modifications
        "modifications": [],
        
        # Supervisor
        "supervisor_decision": "",
        "supervisor_reasoning": "",
        "decision_history": [],
        
        # Loop control
        "iteration_count": 0,
        "max_iterations": max_iterations,
        
        # Output
        "final_report": "",
        "step_logs": []
    }