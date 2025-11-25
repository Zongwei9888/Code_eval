"""
Multi-Agent Repository Analysis Workflow
Clean implementation using LangGraph with proper separation of concerns

Architecture:
+-----------------------------------------------------------------+
|                      WORKFLOW FLOW                               |
+-----------------------------------------------------------------+
|                                                                  |
|   START -> Scanner -> Analyzer -> has_errors? --+-> Reporter     |
|                           ^                     |       |        |
|                           |                     v       v        |
|                           |                   Fixer    END       |
|                           |                     |                |
|                           |                     v                |
|                           |                 Executor             |
|                           |                     |                |
|                           |           +---------+---------+      |
|                           |           |                   |      |
|                           |        success?            failed    |
|                           |           |                   |      |
|                           |           v                   |      |
|                           |       Reporter                |      |
|                           |                               |      |
|                           +-------------------------------+      |
|                            (retry if attempts < max)             |
+-----------------------------------------------------------------+
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List
from datetime import datetime

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm
from agent.state import RepoWorkflowState, create_repo_workflow_state
from prompt.repo_prompts import (
    REPO_SCANNER_SYSTEM_PROMPT,
    REPO_ANALYZER_SYSTEM_PROMPT,
    REPO_FIXER_SYSTEM_PROMPT,
    REPO_EXECUTOR_SYSTEM_PROMPT,
    REPO_REPORTER_SYSTEM_PROMPT,
    format_scan_prompt,
    format_analyze_prompt,
    format_fix_prompt,
    format_execute_prompt
)
from tools.repo_tools import (
    ALL_REPO_TOOLS,
    SCANNER_TOOLS,
    ANALYZER_TOOLS,
    FIXER_TOOLS,
    EXECUTOR_TOOLS
)


# ============================================================================
# AGENT EXECUTOR - Creates multi-turn agent with tool calling
# ============================================================================

def create_agent_executor(
    llm,
    tools: List,
    system_prompt: str,
    agent_name: str,
    max_turns: int = 10
):
    """
    Create an agent that can have multi-turn conversations with tool use.
    Uses a ReAct-style loop: Think -> Act -> Observe -> (repeat)
    
    Args:
        llm: Language model instance
        tools: List of tools available to the agent
        system_prompt: System prompt for the agent
        agent_name: Name for logging
        max_turns: Maximum conversation turns
        
    Returns:
        Agent executor function
    """
    has_tools = bool(tools)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    def agent_executor(state: RepoWorkflowState) -> Dict[str, Any]:
        messages = list(state["messages"])
        step_logs = list(state.get("step_logs", []))
        
        # Add system prompt
        agent_messages = [SystemMessage(content=system_prompt)]
        
        # Add relevant context from history
        for msg in messages[-20:]:
            if not has_tools:
                # Skip ToolMessage for agents without tools
                if isinstance(msg, ToolMessage):
                    continue
                # Skip AIMessage with tool_calls
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    if msg.content:
                        agent_messages.append(AIMessage(content=msg.content))
                    continue
            agent_messages.append(msg)
        
        turn = 0
        final_response = None
        
        while turn < max_turns:
            turn += 1
            
            # Call LLM
            response = llm_with_tools.invoke(agent_messages)
            agent_messages.append(response)
            
            # Log response
            content = response.content if hasattr(response, 'content') else str(response)
            safe_content = content.encode('ascii', 'ignore').decode('ascii')
            
            log_entry = {
                "agent": agent_name,
                "turn": turn,
                "type": "llm_response",
                "content": safe_content[:500],
                "has_tool_calls": bool(response.tool_calls) if hasattr(response, 'tool_calls') else False,
                "timestamp": datetime.now().isoformat()
            }
            step_logs.append(log_entry)
            
            print(f"\n[{agent_name}] Turn {turn}:")
            print(f"  Response: {safe_content[:200]}...")
            
            # Check if agent wants to use tools
            if not response.tool_calls:
                final_response = response
                print(f"  [Done - no more tool calls]")
                break
            
            # Execute tool calls
            print(f"  Tool calls: {[tc['name'] for tc in response.tool_calls]}")
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find and execute tool
                result = f"Tool {tool_name} not found"
                for t in tools:
                    if t.name == tool_name:
                        try:
                            result = t.invoke(tool_args)
                            safe_result = str(result).encode('ascii', 'ignore').decode('ascii')
                            print(f"    {tool_name}: {safe_result[:100]}...")
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            print(f"    {tool_name}: ERROR - {str(e)}")
                        break
                
                # Add tool result to conversation
                agent_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                
                step_logs.append({
                    "agent": agent_name,
                    "turn": turn,
                    "type": "tool_call",
                    "tool": tool_name,
                    "result": str(result)[:300],
                    "timestamp": datetime.now().isoformat()
                })
        
        # Return updated state
        new_messages = agent_messages[1:]  # Exclude system prompt
        
        return {
            "messages": new_messages,
            "step_logs": step_logs
        }
    
    return agent_executor


# ============================================================================
# MULTI-AGENT WORKFLOW
# ============================================================================

class MultiAgentRepoWorkflow:
    """
    Multi-agent workflow for repository analysis with proper feedback loops.
    
    Flow:
    1. Scanner -> Find files
    2. Analyzer -> Check all files for errors
    3. If errors:
       a. Fixer -> Fix the code
       b. Executor -> Run the code
       c. If failed and attempts < max: Go back to Fixer
       d. If success or max attempts: Go to Reporter
    4. Reporter -> Generate final report
    """
    
    def __init__(self, llm_provider: str = "openrouter", max_fix_attempts: int = 5):
        """
        Initialize the multi-agent workflow.
        
        Args:
            llm_provider: LLM provider to use
            max_fix_attempts: Maximum fix attempts per file
        """
        self.llm_provider = llm_provider
        self.max_fix_attempts = max_fix_attempts
        
        # Create LLMs
        self.llm = get_llm(llm_provider, "default")
        self.fast_llm = get_llm(llm_provider, "fast")
        self.powerful_llm = get_llm(llm_provider, "powerful")
        
        # Memory saver for conversation persistence
        self.memory = MemorySaver()
        
        # Build workflow
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph with feedback loops"""
        
        workflow = StateGraph(RepoWorkflowState)
        
        # Create agent executors
        scanner = create_agent_executor(
            self.fast_llm, ALL_REPO_TOOLS, REPO_SCANNER_SYSTEM_PROMPT, "Scanner", max_turns=3
        )
        analyzer = create_agent_executor(
            self.llm, ALL_REPO_TOOLS, REPO_ANALYZER_SYSTEM_PROMPT, "Analyzer", max_turns=15
        )
        fixer = create_agent_executor(
            self.powerful_llm, ALL_REPO_TOOLS, REPO_FIXER_SYSTEM_PROMPT, "Fixer", max_turns=8
        )
        executor = create_agent_executor(
            self.fast_llm, ALL_REPO_TOOLS, REPO_EXECUTOR_SYSTEM_PROMPT, "Executor", max_turns=3
        )
        reporter = create_agent_executor(
            self.llm, [], REPO_REPORTER_SYSTEM_PROMPT, "Reporter", max_turns=2
        )
        
        # Add nodes
        workflow.add_node("scanner", self._scanner_wrapper(scanner))
        workflow.add_node("analyzer", self._analyzer_wrapper(analyzer))
        workflow.add_node("fixer", self._fixer_wrapper(fixer))
        workflow.add_node("executor", self._executor_wrapper(executor))
        workflow.add_node("reporter", reporter)
        
        # Define edges
        workflow.add_edge(START, "scanner")
        workflow.add_edge("scanner", "analyzer")
        
        # After analyzer: check if there are errors to fix
        workflow.add_conditional_edges(
            "analyzer",
            self._has_errors,
            {
                "has_errors": "fixer",
                "no_errors": "reporter"
            }
        )
        
        # After fixer: execute to verify
        workflow.add_edge("fixer", "executor")
        
        # After executor: check result and decide next step
        workflow.add_conditional_edges(
            "executor",
            self._execution_decision,
            {
                "success": "reporter",
                "retry": "fixer",
                "give_up": "reporter"
            }
        )
        
        workflow.add_edge("reporter", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _scanner_wrapper(self, scanner_fn):
        """Wrapper to process scanner results"""
        def wrapper(state: RepoWorkflowState) -> Dict[str, Any]:
            project_path = state["project_path"]
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=format_scan_prompt(project_path))
            ]
            
            result = scanner_fn(state)
            
            # Parse scan results from messages
            python_files = []
            test_files = []
            
            for msg in result.get("messages", []):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if "python_files" in content:
                    try:
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start >= 0 and end > start:
                            data = json.loads(content[start:end])
                            python_files = data.get("python_files", [])
                            test_files = data.get("test_files", [])
                    except:
                        pass
            
            result["python_files"] = python_files
            result["test_files"] = test_files
            
            return result
        
        return wrapper
    
    def _analyzer_wrapper(self, analyzer_fn):
        """Wrapper to process analyzer results"""
        def wrapper(state: RepoWorkflowState) -> Dict[str, Any]:
            files = state.get("python_files", [])
            project_path = state["project_path"]
            
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=format_analyze_prompt(project_path, files))
            ]
            
            result = analyzer_fn(state)
            
            # Parse for files with errors
            files_with_errors = []
            for msg in result.get("messages", []):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if '"valid": false' in content.lower() or '"valid":false' in content.lower():
                    try:
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start >= 0 and end > start:
                            data = json.loads(content[start:end])
                            if not data.get("valid", True):
                                files_with_errors.append({
                                    "file": data.get("file", "unknown"),
                                    "error": data.get("error", "unknown error"),
                                    "line": data.get("line_number")
                                })
                    except:
                        pass
            
            result["files_with_errors"] = files_with_errors
            if files_with_errors:
                result["current_file"] = files_with_errors[0]["file"]
            
            return result
        
        return wrapper
    
    def _fixer_wrapper(self, fixer_fn):
        """Wrapper to track fix attempts"""
        def wrapper(state: RepoWorkflowState) -> Dict[str, Any]:
            current_file = state.get("current_file")
            errors = state.get("files_with_errors", [])
            attempt = state.get("fix_attempts", 0) + 1
            
            # Find current error
            error_info = "Unknown error"
            for err in errors:
                if err.get("file") == current_file:
                    error_info = err.get("error", "Unknown error")
                    break
            
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=format_fix_prompt(attempt, current_file, error_info))
            ]
            
            result = fixer_fn(state)
            result["fix_attempts"] = attempt
            
            # Track fix
            fixes = list(state.get("fixes_applied", []))
            fixes.append({
                "file": current_file,
                "attempt": attempt,
                "timestamp": datetime.now().isoformat()
            })
            result["fixes_applied"] = fixes
            
            return result
        
        return wrapper
    
    def _executor_wrapper(self, executor_fn):
        """Wrapper to track execution results"""
        def wrapper(state: RepoWorkflowState) -> Dict[str, Any]:
            current_file = state.get("current_file")
            project_path = state["project_path"]
            
            # Build full path
            if current_file and not Path(current_file).is_absolute():
                full_path = str(Path(project_path) / current_file)
            else:
                full_path = current_file or project_path
            
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=format_execute_prompt(full_path))
            ]
            
            result = executor_fn(state)
            
            # Parse execution result
            success = False
            for msg in reversed(result.get("messages", [])[-5:]):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if '"success": true' in content.lower() or '"success":true' in content.lower():
                    success = True
                    break
                if '"success": false' in content.lower() or '"success":false' in content.lower():
                    success = False
                    break
            
            result["last_execution_success"] = success
            
            # Track result
            exec_results = list(state.get("execution_results", []))
            exec_results.append({
                "file": current_file,
                "success": success,
                "attempt": state.get("fix_attempts", 0),
                "timestamp": datetime.now().isoformat()
            })
            result["execution_results"] = exec_results
            
            return result
        
        return wrapper
    
    def _has_errors(self, state: RepoWorkflowState) -> Literal["has_errors", "no_errors"]:
        """Check if there are files with errors"""
        errors = state.get("files_with_errors", [])
        
        if errors:
            print(f"\n[Decision] Found {len(errors)} file(s) with errors -> Go to Fixer")
            return "has_errors"
        else:
            print(f"\n[Decision] No errors found -> Go to Reporter")
            return "no_errors"
    
    def _execution_decision(self, state: RepoWorkflowState) -> Literal["success", "retry", "give_up"]:
        """Decide what to do after execution"""
        success = state.get("last_execution_success", False)
        attempts = state.get("fix_attempts", 0)
        max_attempts = state.get("max_fix_attempts", self.max_fix_attempts)
        
        if success:
            print(f"\n[Decision] Execution SUCCESS -> Go to Reporter")
            return "success"
        elif attempts < max_attempts:
            print(f"\n[Decision] Execution FAILED (attempt {attempts}/{max_attempts}) -> Retry Fixer")
            return "retry"
        else:
            print(f"\n[Decision] Max attempts ({max_attempts}) reached -> Go to Reporter")
            return "give_up"
    
    def run(self, project_path: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the workflow on a project.
        
        Args:
            project_path: Path to the project directory
            thread_id: Optional thread ID for session management
            
        Returns:
            Final state dictionary
        """
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        initial_state = create_repo_workflow_state(project_path, self.max_fix_attempts)
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*70}")
        print(f"  MULTI-AGENT REPOSITORY ANALYSIS WORKFLOW")
        print(f"{'='*70}")
        print(f"  Project: {project_path}")
        print(f"  Thread:  {thread_id}")
        print(f"  Max Fix Attempts: {self.max_fix_attempts}")
        print(f"{'='*70}\n")
        
        return self.graph.invoke(initial_state, config)
    
    def stream_run(self, project_path: str, thread_id: Optional[str] = None):
        """
        Stream the workflow execution.
        
        Args:
            project_path: Path to the project directory
            thread_id: Optional thread ID for session management
            
        Yields:
            State updates as they occur
        """
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        initial_state = create_repo_workflow_state(project_path, self.max_fix_attempts)
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_multi_agent_workflow(
    llm_provider: str = "openrouter",
    max_fix_attempts: int = 5
) -> MultiAgentRepoWorkflow:
    """
    Factory function to create the multi-agent workflow.
    
    Args:
        llm_provider: LLM provider to use
        max_fix_attempts: Maximum fix attempts per file
        
    Returns:
        MultiAgentRepoWorkflow instance
    """
    return MultiAgentRepoWorkflow(llm_provider, max_fix_attempts)

