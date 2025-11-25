"""
LangGraph workflow with memory/checkpointing support
Implements best practices from LangGraph documentation
"""
from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteSaver
except ImportError:
    SqliteSaver = None
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import os

from agent import (
    MultiAgentState,
    CodeAnalyzerAgent,
    CodeExecutorAgent,
    CodeModifierAgent,
    create_agents,
    create_initial_state
)
from config import MAX_EXECUTION_ATTEMPTS
from prompt import format_workflow_start_prompt


class CodeImprovementWorkflow:
    """
    Multi-agent workflow for code improvement with memory/persistence
    
    Features:
    - Memory/Checkpointing: Maintains state across invocations
    - Thread ID support: Multiple concurrent sessions
    - Streaming: Real-time updates
    - Proper LangGraph patterns: Following official documentation
    
    Flow:
    1. START -> analyze_code: Analyze code structure and issues
    2. analyze_code -> execute_code: Execute code to verify
    3. execute_code -> check_success: Check if execution succeeded
    4. check_success -> END (if success or max attempts)
    5. check_success -> modify_code (if failed and attempts remaining)
    6. modify_code -> analyze_code: Re-analyze after modification
    """
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        max_attempts: int = MAX_EXECUTION_ATTEMPTS,
        use_sqlite: bool = False,
        sqlite_path: str = "checkpoints.db"
    ):
        """
        Initialize workflow with memory support
        
        Args:
            llm_provider: LLM provider to use
            max_attempts: Maximum execution attempts
            use_sqlite: Use SQLite for persistent checkpointing
            sqlite_path: Path to SQLite database file
        """
        self.llm_provider = llm_provider
        self.max_attempts = max_attempts
        self.agents = create_agents(llm_provider)
        
        # Initialize checkpointer based on preference
        if use_sqlite and SqliteSaver:
            # Production: SQLite persistence
            self.checkpointer = SqliteSaver.from_conn_string(sqlite_path)
            print(f"  [+] Using SQLite checkpointer: {sqlite_path}")
        else:
            # Development: In-memory checkpointing
            self.checkpointer = MemorySaver()
            print("  [+] Using in-memory checkpointer")
        
        # Build graph with checkpointer
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow with proper state management"""
        
        # Create state graph with proper state type
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes for each agent
        workflow.add_node("analyze_code", self._analyze_node)
        workflow.add_node("execute_code", self._execute_node)
        workflow.add_node("modify_code", self._modify_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.add_edge(START, "analyze_code")
        workflow.add_edge("analyze_code", "execute_code")
        
        # Conditional edge based on execution result
        workflow.add_conditional_edges(
            "execute_code",
            self._should_continue,
            {
                "modify": "modify_code",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("modify_code", "analyze_code")
        workflow.add_edge("finalize", END)
        
        # Compile with checkpointer for memory support
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _analyze_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code analysis"""
        print(f"\n{'='*60}")
        print(f"  [ANALYZE] Code Analysis (Attempt {state['execution_attempts'] + 1}/{self.max_attempts})")
        print(f"{'='*60}")
        
        analyzer = self.agents["analyzer"]
        result = analyzer.analyze(state)
        
        # Log LLM response
        if result.get("code_analysis"):
            response_text = result["code_analysis"]
            print(f"\n  [LLM Response]:")
            for line in response_text.split('\n')[:10]:
                print(f"      {line[:80]}")
            if len(response_text.split('\n')) > 10:
                print(f"      ... ({len(response_text.split('\\n')) - 10} more lines)")
        
        print(f"\n  [+] Analysis Complete:")
        print(f"      Issues found: {len(result.get('identified_issues', []))}")
        
        return result
    
    def _execute_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code execution"""
        print(f"\n{'='*60}")
        print(f"  [EXECUTE] Running Code")
        print(f"{'='*60}")
        
        executor = self.agents["executor"]
        result = executor.execute(state)
        
        success = result.get("execution_success", False)
        print(f"\n  [{'+'if success else '!'}] Execution {'SUCCEEDED' if success else 'FAILED'}")
        
        if not success and result.get("last_error"):
            print(f"\n  [!] Error Details:")
            error_lines = result["last_error"].split('\n')[:5]
            for line in error_lines:
                print(f"      {line}")
        
        return result
    
    def _modify_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code modification"""
        print(f"\n{'='*60}")
        print(f"  [MODIFY] Applying Fixes")
        print(f"{'='*60}")
        
        modifier = self.agents["modifier"]
        result = modifier.modify(state)
        
        print(f"\n  [+] Modifications applied")
        print(f"      Changes tracked in history")
        
        return result
    
    def _finalize_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for finalizing results"""
        print(f"\n{'='*60}")
        print(f"  [FINALIZE] Results Summary")
        print(f"{'='*60}")
        
        if state["execution_success"]:
            status = f"[+] SUCCESS: Code executes correctly after {state['execution_attempts']} attempt(s)"
        else:
            status = f"[!] INCOMPLETE: Max attempts ({self.max_attempts}) reached. Code still has issues."
        
        print(f"\n  {status}")
        print(f"\n  Summary:")
        print(f"      Total attempts:    {state['execution_attempts']}")
        print(f"      Modifications:     {len(state['modification_history'])}")
        print(f"      Final status:      {'Success' if state['execution_success'] else 'Needs more work'}")
        
        return {
            "final_status": status,
            "should_continue": False
        }
    
    def _should_continue(self, state: MultiAgentState) -> Literal["modify", "finalize"]:
        """
        Decide next step based on execution results
        
        Returns:
            - "modify": Continue to modification (if failed and attempts remain)
            - "finalize": Finalize results
        """
        
        # Check if execution succeeded
        if state["execution_success"]:
            return "finalize"
        
        # Check if max attempts reached
        if state["execution_attempts"] >= self.max_attempts:
            return "finalize"
        
        # Continue trying to fix
        return "modify"
    
    def run(
        self,
        file_path: str,
        initial_code: str = "",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the workflow on a code file with memory support
        
        Args:
            file_path: Path to the code file
            initial_code: Initial code content (optional, will read from file if not provided)
            thread_id: Thread ID for session management (optional, generated if not provided)
            
        Returns:
            Final state dictionary
        """
        # Generate thread ID if not provided
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"  CODE IMPROVEMENT WORKFLOW")
        print(f"{'='*80}")
        print(f"  Target File:   {file_path}")
        print(f"  Max Attempts:  {self.max_attempts}")
        print(f"  LLM Provider:  {self.llm_provider}")
        print(f"  Thread ID:     {thread_id}")
        
        # Create config with thread ID for checkpointing
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Initialize state with messages
        start_prompt = format_workflow_start_prompt(file_path)
        initial_state = create_initial_state(
            target_file=file_path,
            initial_code=initial_code,
            max_attempts=self.max_attempts
        )
        initial_state["messages"] = [HumanMessage(content=start_prompt)]
        
        # Run workflow with config
        try:
            final_state = self.graph.invoke(initial_state, config)
            
            print(f"\n{'='*80}")
            print(f"  WORKFLOW COMPLETE")
            print(f"{'='*80}")
            
            return final_state
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"  [ERROR] Workflow Error: {str(e)}")
            print(f"{'='*80}")
            raise
    
    def stream_run(
        self,
        file_path: str,
        initial_code: str = "",
        thread_id: Optional[str] = None
    ):
        """
        Run workflow with streaming output and memory support
        
        Args:
            file_path: Path to the code file
            initial_code: Initial code content
            thread_id: Thread ID for session management
            
        Yields:
            State updates as they occur
        """
        # Generate thread ID if not provided
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"  CODE IMPROVEMENT WORKFLOW (Streaming)")
        print(f"{'='*80}")
        print(f"  Thread ID: {thread_id}")
        
        # Create config with thread ID
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Initialize state
        start_prompt = format_workflow_start_prompt(file_path)
        initial_state = create_initial_state(
            target_file=file_path,
            initial_code=initial_code,
            max_attempts=self.max_attempts
        )
        initial_state["messages"] = [HumanMessage(content=start_prompt)]
        
        # Stream workflow updates
        for state_update in self.graph.stream(initial_state, config):
            yield state_update
    
    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state for a thread
        
        Args:
            thread_id: Thread ID to retrieve state for
            
        Returns:
            Current state or None if not found
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        try:
            state_snapshot = self.graph.get_state(config)
            return state_snapshot.values if state_snapshot else None
        except Exception as e:
            print(f"⚠️  Could not retrieve state: {str(e)}")
            return None
    
    def resume(self, thread_id: str) -> Dict[str, Any]:
        """
        Resume workflow from checkpoint
        
        Args:
            thread_id: Thread ID to resume
            
        Returns:
            Final state after resuming
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        print(f"\n{'='*80}")
        print(f"  RESUMING WORKFLOW")
        print(f"{'='*80}")
        print(f"  Thread ID: {thread_id}")
        
        try:
            # Resume from checkpoint (pass None to continue from last state)
            final_state = self.graph.invoke(None, config)
            
            print(f"\n{'='*80}")
            print(f"  WORKFLOW RESUMED AND COMPLETED")
            print(f"{'='*80}")
            
            return final_state
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"  [ERROR] Resume Error: {str(e)}")
            print(f"{'='*80}")
            raise
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Visualize the workflow graph
        
        Args:
            output_path: Path to save visualization
        """
        try:
            from IPython.display import Image, display
            
            # Generate graph visualization
            graph_image = self.graph.get_graph().draw_mermaid_png()
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(graph_image)
            
            print(f"  [+] Workflow graph saved to: {output_path}")
            
            return Image(graph_image)
            
        except ImportError:
            print("  [!] IPython not available. Cannot display graph.")
            print("      Install with: pip install ipython")
        except Exception as e:
            print(f"  [!] Could not generate graph: {str(e)}")


def create_workflow(
    llm_provider: str = "openrouter",
    max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    use_sqlite: bool = False,
    sqlite_path: str = "checkpoints.db"
):
    """
    Factory function to create workflow with memory support
    
    Args:
        llm_provider: LLM provider to use
        max_attempts: Maximum execution attempts
        use_sqlite: Use SQLite for persistent checkpointing
        sqlite_path: Path to SQLite database file
        
    Returns:
        CodeImprovementWorkflow instance
    """
    return CodeImprovementWorkflow(
        llm_provider=llm_provider,
        max_attempts=max_attempts,
        use_sqlite=use_sqlite,
        sqlite_path=sqlite_path
    )

