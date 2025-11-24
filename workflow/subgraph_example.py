"""
Example of using subgraphs for complex multi-file code improvement
Demonstrates LangGraph subgraph pattern from official documentation
"""
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, HumanMessage

from agent import create_agents
from agent.state import MultiAgentState, create_initial_state
from config import MAX_EXECUTION_ATTEMPTS


class MultiFileState(TypedDict):
    """
    State for multi-file improvement workflow
    Coordinates improvement of multiple files
    """
    messages: Annotated[List[AnyMessage], add_messages]
    files_to_process: List[str]
    current_file_index: int
    file_results: Dict[str, Any]
    overall_success: bool


def create_file_improvement_subgraph(llm_provider: str = "openrouter"):
    """
    Create a subgraph for improving a single file
    This encapsulates the analyze -> execute -> modify loop
    
    Args:
        llm_provider: LLM provider to use
        
    Returns:
        Compiled subgraph for file improvement
    """
    agents = create_agents(llm_provider)
    
    # Create subgraph with same state type as main workflow
    subgraph = StateGraph(MultiAgentState)
    
    # Define nodes
    def analyze_node(state: MultiAgentState) -> Dict[str, Any]:
        """Analyze code node"""
        print(f"  🔍 Analyzing: {state['target_file']}")
        return agents["analyzer"].analyze(state)
    
    def execute_node(state: MultiAgentState) -> Dict[str, Any]:
        """Execute code node"""
        print(f"  ▶️  Executing: {state['target_file']}")
        return agents["executor"].execute(state)
    
    def modify_node(state: MultiAgentState) -> Dict[str, Any]:
        """Modify code node"""
        print(f"  🔧 Modifying: {state['target_file']}")
        return agents["modifier"].modify(state)
    
    def should_continue(state: MultiAgentState) -> str:
        """Decide whether to continue or finish"""
        if state["execution_success"]:
            return "end"
        if state["execution_attempts"] >= state["max_attempts"]:
            return "end"
        return "modify"
    
    # Add nodes to subgraph
    subgraph.add_node("analyze", analyze_node)
    subgraph.add_node("execute", execute_node)
    subgraph.add_node("modify", modify_node)
    
    # Define edges
    subgraph.add_edge(START, "analyze")
    subgraph.add_edge("analyze", "execute")
    subgraph.add_conditional_edges(
        "execute",
        should_continue,
        {
            "modify": "modify",
            "end": END
        }
    )
    subgraph.add_edge("modify", "analyze")
    
    return subgraph.compile()


def create_multi_file_workflow(llm_provider: str = "openrouter"):
    """
    Create a workflow that processes multiple files using subgraphs
    
    Demonstrates:
    - Using compiled subgraph as a node
    - Coordinating multiple file improvements
    - State transformation between main graph and subgraph
    
    Args:
        llm_provider: LLM provider to use
        
    Returns:
        Compiled main workflow
    """
    # Create the file improvement subgraph
    file_subgraph = create_file_improvement_subgraph(llm_provider)
    
    # Create main workflow
    main_workflow = StateGraph(MultiFileState)
    
    def initialize_node(state: MultiFileState) -> Dict[str, Any]:
        """Initialize processing"""
        print(f"\n🚀 Starting multi-file improvement")
        print(f"   Files to process: {len(state['files_to_process'])}")
        return {
            "current_file_index": 0,
            "file_results": {},
            "overall_success": True
        }
    
    def process_file_node(state: MultiFileState) -> Dict[str, Any]:
        """Process current file using subgraph"""
        current_idx = state["current_file_index"]
        file_path = state["files_to_process"][current_idx]
        
        print(f"\n📁 Processing file {current_idx + 1}/{len(state['files_to_process'])}: {file_path}")
        
        # Create state for subgraph
        file_state = create_initial_state(
            target_file=file_path,
            initial_code="",  # Will be read by analyzer
            max_attempts=MAX_EXECUTION_ATTEMPTS
        )
        file_state["messages"] = [HumanMessage(content=f"Improve {file_path}")]
        
        # Invoke subgraph
        result = file_subgraph.invoke(file_state)
        
        # Store result
        file_results = state["file_results"].copy()
        file_results[file_path] = {
            "success": result.get("execution_success", False),
            "attempts": result.get("execution_attempts", 0),
            "final_status": result.get("final_status", "")
        }
        
        return {
            "file_results": file_results,
            "overall_success": state["overall_success"] and result.get("execution_success", False)
        }
    
    def move_to_next_file_node(state: MultiFileState) -> Dict[str, Any]:
        """Move to next file"""
        return {
            "current_file_index": state["current_file_index"] + 1
        }
    
    def should_process_more_files(state: MultiFileState) -> str:
        """Check if there are more files to process"""
        if state["current_file_index"] >= len(state["files_to_process"]):
            return "finalize"
        return "process_file"
    
    def finalize_node(state: MultiFileState) -> Dict[str, Any]:
        """Finalize multi-file processing"""
        print(f"\n{'='*60}")
        print(f"🏁 Multi-file processing complete")
        print(f"{'='*60}")
        print(f"Files processed: {len(state['file_results'])}")
        print(f"Overall success: {'✅ Yes' if state['overall_success'] else '❌ No'}")
        
        for file_path, result in state["file_results"].items():
            status_emoji = "✅" if result["success"] else "❌"
            print(f"  {status_emoji} {file_path}: {result['attempts']} attempt(s)")
        
        return {}
    
    # Add nodes to main workflow
    main_workflow.add_node("initialize", initialize_node)
    main_workflow.add_node("process_file", process_file_node)
    main_workflow.add_node("move_to_next", move_to_next_file_node)
    main_workflow.add_node("finalize", finalize_node)
    
    # Define edges
    main_workflow.add_edge(START, "initialize")
    main_workflow.add_edge("initialize", "process_file")
    main_workflow.add_edge("process_file", "move_to_next")
    main_workflow.add_conditional_edges(
        "move_to_next",
        should_process_more_files,
        {
            "process_file": "process_file",
            "finalize": "finalize"
        }
    )
    main_workflow.add_edge("finalize", END)
    
    return main_workflow.compile()


# Example usage
if __name__ == "__main__":
    """
    Example: Process multiple files using subgraphs
    """
    print("="*80)
    print("Multi-File Improvement Example (with Subgraphs)")
    print("="*80)
    
    # Create workflow
    workflow = create_multi_file_workflow("openrouter")
    
    # Example: Process multiple files
    files = [
        "script1.py",
        "script2.py",
        "script3.py"
    ]
    
    initial_state: MultiFileState = {
        "messages": [HumanMessage(content="Improve all files")],
        "files_to_process": files,
        "current_file_index": 0,
        "file_results": {},
        "overall_success": True
    }
    
    # Note: This is just a structure example
    # Uncomment to actually run:
    # result = workflow.invoke(initial_state)
    
    print("\n📝 Note: This is a structure example.")
    print("   Uncomment the invoke line to actually run the workflow.")
    print("\n✅ Subgraph pattern demonstrated successfully!")

