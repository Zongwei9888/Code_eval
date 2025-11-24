"""
LangGraph workflow orchestrating multi-agent code improvement system
Implements iterative analyze -> execute -> modify loop until code works
"""
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from agent import (
    MultiAgentState,
    CodeAnalyzerAgent,
    CodeExecutorAgent,
    CodeModifierAgent,
    create_agents
)
from config import MAX_EXECUTION_ATTEMPTS
from prompt import format_workflow_start_prompt


class CodeImprovementWorkflow:
    """
    Multi-agent workflow for code improvement
    
    Flow:
    1. START -> analyze_code: Analyze code structure and issues
    2. analyze_code -> execute_code: Execute code to verify
    3. execute_code -> check_success: Check if execution succeeded
    4. check_success -> END (if success or max attempts)
    5. check_success -> modify_code (if failed and attempts remaining)
    6. modify_code -> analyze_code: Re-analyze after modification
    """
    
    def __init__(self, llm_provider: str = "openrouter", max_attempts: int = MAX_EXECUTION_ATTEMPTS):
        self.llm_provider = llm_provider
        self.max_attempts = max_attempts
        self.agents = create_agents(llm_provider)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create state graph
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
                "finalize": "finalize",
                "end": END
            }
        )
        
        workflow.add_edge("modify_code", "analyze_code")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _analyze_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code analysis"""
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING CODE (Attempt {state.execution_attempts + 1}/{self.max_attempts})")
        print(f"{'='*60}")
        
        analyzer = self.agents["analyzer"]
        result = analyzer.analyze(state)
        
        print(f"\n📊 Analysis Complete:")
        print(f"   - Issues found: {len(result.get('identified_issues', []))}")
        
        return result
    
    def _execute_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code execution"""
        print(f"\n{'='*60}")
        print(f"▶️  EXECUTING CODE")
        print(f"{'='*60}")
        
        executor = self.agents["executor"]
        result = executor.execute(state)
        
        success = result.get("execution_success", False)
        print(f"\n✅ Execution {'SUCCEEDED' if success else '❌ FAILED'}")
        
        if not success and result.get("last_error"):
            print(f"\n⚠️  Error Details:")
            error_lines = result["last_error"].split('\n')[:5]
            for line in error_lines:
                print(f"   {line}")
        
        return result
    
    def _modify_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for code modification"""
        print(f"\n{'='*60}")
        print(f"🔧 MODIFYING CODE")
        print(f"{'='*60}")
        
        modifier = self.agents["modifier"]
        result = modifier.modify(state)
        
        print(f"\n✏️  Modifications applied")
        print(f"   - Changes tracked in history")
        
        return result
    
    def _finalize_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Node for finalizing results"""
        print(f"\n{'='*60}")
        print(f"🎯 FINALIZING RESULTS")
        print(f"{'='*60}")
        
        if state.execution_success:
            status = f"✅ SUCCESS: Code executes correctly after {state.execution_attempts} attempt(s)"
        else:
            status = f"⚠️  INCOMPLETE: Max attempts ({self.max_attempts}) reached. Code still has issues."
        
        print(f"\n{status}")
        print(f"\n📋 Summary:")
        print(f"   - Total attempts: {state.execution_attempts}")
        print(f"   - Modifications made: {len(state.modification_history)}")
        print(f"   - Final status: {'Success' if state.execution_success else 'Needs more work'}")
        
        return {
            "final_status": status,
            "should_continue": False
        }
    
    def _should_continue(self, state: MultiAgentState) -> Literal["modify", "finalize", "end"]:
        """
        Decide next step based on execution results
        
        Returns:
            - "modify": Continue to modification (if failed and attempts remain)
            - "finalize": Finalize results
            - "end": End immediately (not used currently)
        """
        
        # Check if execution succeeded
        if state.execution_success:
            return "finalize"
        
        # Check if max attempts reached
        if state.execution_attempts >= self.max_attempts:
            return "finalize"
        
        # Continue trying to fix
        return "modify"
    
    def run(self, file_path: str, initial_code: str = "") -> Dict[str, Any]:
        """
        Run the workflow on a code file
        
        Args:
            file_path: Path to the code file
            initial_code: Initial code content (optional, will read from file if not provided)
            
        Returns:
            Final state dictionary
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Code Improvement Workflow")
        print(f"{'='*80}")
        print(f"Target File: {file_path}")
        print(f"Max Attempts: {self.max_attempts}")
        print(f"LLM Provider: {self.llm_provider}")
        
        # Initialize state
        start_prompt = format_workflow_start_prompt(file_path)
        initial_state = MultiAgentState(
            messages=[
                HumanMessage(content=start_prompt)
            ],
            target_file=file_path,
            file_content=initial_code,
            current_code=initial_code,
            max_attempts=self.max_attempts
        )
        
        # Run workflow
        try:
            final_state = self.graph.invoke(initial_state)
            
            print(f"\n{'='*80}")
            print(f"🏁 Workflow Complete")
            print(f"{'='*80}")
            
            return final_state
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ Workflow Error: {str(e)}")
            print(f"{'='*80}")
            raise
    
    def stream_run(self, file_path: str, initial_code: str = ""):
        """
        Run workflow with streaming output
        
        Args:
            file_path: Path to the code file
            initial_code: Initial code content
            
        Yields:
            State updates as they occur
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Code Improvement Workflow (Streaming)")
        print(f"{'='*80}")
        
        start_prompt = format_workflow_start_prompt(file_path)
        initial_state = MultiAgentState(
            messages=[
                HumanMessage(content=start_prompt)
            ],
            target_file=file_path,
            file_content=initial_code,
            current_code=initial_code,
            max_attempts=self.max_attempts
        )
        
        for state_update in self.graph.stream(initial_state):
            yield state_update
    
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
            
            print(f"✅ Workflow graph saved to: {output_path}")
            
            return Image(graph_image)
            
        except ImportError:
            print("⚠️  IPython not available. Cannot display graph.")
            print("   Install with: pip install ipython")
        except Exception as e:
            print(f"⚠️  Could not generate graph: {str(e)}")


def create_workflow(llm_provider: str = "openrouter", max_attempts: int = MAX_EXECUTION_ATTEMPTS):
    """
    Factory function to create workflow
    
    Args:
        llm_provider: LLM provider to use
        max_attempts: Maximum execution attempts
        
    Returns:
        CodeImprovementWorkflow instance
    """
    return CodeImprovementWorkflow(llm_provider=llm_provider, max_attempts=max_attempts)


# Interactive mode functions
def interactive_session(llm_provider: str = "openrouter"):
    """
    Start an interactive session with the workflow
    Allows user to submit files and see results in real-time
    """
    print(f"\n{'='*80}")
    print(f"🤖 Multi-Agent Code Assistant - Interactive Mode")
    print(f"{'='*80}")
    print(f"\nLLM Provider: {llm_provider}")
    print(f"\nCommands:")
    print(f"  - Enter a file path to analyze and fix")
    print(f"  - Type 'quit' or 'exit' to end session")
    print(f"  - Type 'help' for more information")
    print(f"\n{'='*80}\n")
    
    workflow = create_workflow(llm_provider)
    
    while True:
        try:
            user_input = input("\n📁 Enter file path (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n📖 Help:")
                print("  - Provide a path to a Python file")
                print("  - The system will analyze, execute, and fix the code")
                print("  - Results are saved back to the file")
                print("  - Maximum 5 attempts per file")
                continue
            
            # Check if file exists
            from pathlib import Path
            file_path = Path(user_input)
            
            if not file_path.exists():
                print(f"❌ File not found: {user_input}")
                create = input("Create new file? (y/n): ").strip().lower()
                if create == 'y':
                    code = input("Enter initial code (or press Enter for empty): ").strip()
                    if code:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(code)
                    else:
                        continue
                else:
                    continue
            
            # Read initial code
            initial_code = file_path.read_text()
            
            # Run workflow
            result = workflow.run(str(file_path), initial_code)
            
            print(f"\n{'='*80}")
            if result.get("execution_success"):
                print(f"✅ SUCCESS! Code is now working correctly.")
            else:
                print(f"⚠️  Could not fully fix the code. Please review manually.")
            print(f"{'='*80}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run interactive session
    interactive_session()

