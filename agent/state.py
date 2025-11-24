"""
State definitions for multi-agent code improvement workflow
Following LangGraph best practices with proper TypedDict usage
"""
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class MultiAgentState(TypedDict):
    """
    Shared state across all agents
    Uses TypedDict for proper LangGraph integration
    
    Key features:
    - messages: Annotated with add_messages for proper message handling
    - All fields properly typed for LangGraph state management
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


def create_initial_state(
    target_file: str,
    initial_code: str = "",
    max_attempts: int = 5
) -> MultiAgentState:
    """
    Create initial state with default values
    
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

