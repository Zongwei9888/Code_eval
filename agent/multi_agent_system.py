"""
Multi-Agent Code Analysis System with Proper Feedback Loops
Based on LangGraph best practices:
- https://docs.langchain.com/oss/python/langgraph/workflows-agents
- https://docs.langchain.com/oss/python/langgraph/add-memory

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      WORKFLOW FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   START ──► Scanner ──► Analyzer ──► has_errors? ──┬──► Reporter │
│                             ▲                      │       │     │
│                             │                      ▼       ▼     │
│                             │                   Fixer    END     │
│                             │                      │             │
│                             │                      ▼             │
│                             │                  Executor          │
│                             │                      │             │
│                             │            ┌────────┴────────┐     │
│                             │            │                 │     │
│                             │         success?          failed   │
│                             │            │                 │     │
│                             │            ▼                 │     │
│                             │        Reporter              │     │
│                             │                              │     │
│                             └──────────────────────────────┘     │
│                              (retry if attempts < max)           │
└─────────────────────────────────────────────────────────────────┘

Key Features:
1. Feedback loop: Fixer -> Executor -> (retry if failed)
2. Memory: Conversation history preserved
3. Clear end conditions: Success OR max_attempts reached
4. Multi-turn conversations per agent
"""
import os
import sys
import ast
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime
import operator

from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm


# ============================================================================
# TOOLS
# ============================================================================

@tool
def scan_project(project_path: str) -> str:
    """
    Scan a project directory to find all Python files.
    Returns a JSON with python_files and test_files.
    """
    root = Path(project_path)
    if not root.exists():
        return json.dumps({"error": f"Path does not exist: {project_path}"})
    
    python_files = []
    test_files = []
    ignore_dirs = {'__pycache__', '.git', 'venv', '.venv', 'node_modules', 'build', 'dist', '.pytest_cache'}
    
    for py_file in root.rglob("*.py"):
        if any(d in py_file.parts for d in ignore_dirs):
            continue
        rel_path = str(py_file.relative_to(root))
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py") or "tests" in py_file.parts:
            test_files.append(rel_path)
        else:
            python_files.append(rel_path)
    
    return json.dumps({
        "project_path": str(root),
        "python_files": sorted(python_files),
        "test_files": sorted(test_files),
        "total_files": len(python_files) + len(test_files)
    }, indent=2)


@tool
def read_file(file_path: str) -> str:
    """Read the content of a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        content = path.read_text(encoding='utf-8')
        return f"=== FILE: {file_path} ===\n=== LINES: {len(content.splitlines())} ===\n\n{content}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Use this to fix code."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"SUCCESS: Written {len(content)} characters to {file_path}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def check_syntax(file_path: str) -> str:
    """Check Python syntax of a file using AST parser."""
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"valid": False, "error": "File not found", "file": file_path})
        
        content = path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            return json.dumps({
                "valid": True,
                "file": file_path,
                "lines": len(content.splitlines()),
                "classes": classes,
                "functions": functions
            }, indent=2)
        except SyntaxError as e:
            return json.dumps({
                "valid": False,
                "file": file_path,
                "error": f"Line {e.lineno}: {e.msg}",
                "line_number": e.lineno,
                "offset": e.offset
            }, indent=2)
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e), "file": file_path})


@tool
def execute_python_file(file_path: str, timeout: int = 30) -> str:
    """
    Execute a Python file and capture stdout/stderr.
    Returns success status, output, and any errors.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": "File not found"})
        
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(path.parent)
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[:3000] if result.stdout else "",
            "stderr": result.stderr[:3000] if result.stderr else "",
            "file": file_path
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Timeout", "file": file_path})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "file": file_path})


@tool
def run_pytest(project_path: str, timeout: int = 120) -> str:
    """Run pytest on a project and return results."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_path
        )
        
        output = result.stdout + result.stderr
        passed = output.count(" PASSED") + output.count(" passed")
        failed = output.count(" FAILED") + output.count(" failed") + output.count(" ERROR")
        
        return json.dumps({
            "success": result.returncode == 0,
            "passed": passed,
            "failed": failed,
            "output": output[:4000]
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Tests timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# All available tools
ALL_TOOLS = [scan_project, read_file, write_file, check_syntax, execute_python_file, run_pytest]


# ============================================================================
# STATE DEFINITION
# ============================================================================

class WorkflowState(TypedDict):
    """
    Shared state for the entire workflow.
    Memory is preserved through messages.
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
# AGENT PROMPTS
# ============================================================================

SCANNER_PROMPT = """You are a Project Scanner Agent.

Your task:
1. Use scan_project tool to find all Python files in the project
2. Report what you found

After scanning, summarize:
- How many Python files found
- How many test files found
- Project structure overview

DO NOT call any other tools. Just scan and summarize."""


ANALYZER_PROMPT = """You are a Code Analyzer Agent.

Your task:
1. Check syntax of EACH Python file using check_syntax tool
2. Read files that have errors to understand the issues
3. Create a detailed report of all issues found

Be THOROUGH - check EVERY file. Call check_syntax for each file one by one.

When done, summarize:
- Total files analyzed
- Files with syntax errors (list them)
- Types of errors found

If no errors found, say "All files have valid syntax"."""


FIXER_PROMPT = """You are a Code Fixer Agent.

Your task:
1. Read the file with errors using read_file tool
2. Understand the error from the analysis
3. Fix the code (make minimal changes)
4. Write the fixed code using write_file tool

IMPORTANT:
- Read the ENTIRE file first
- Make ONLY necessary changes to fix the error
- Preserve all existing functionality
- Write the COMPLETE fixed file (not just the changed parts)

After fixing, confirm what you changed."""


EXECUTOR_PROMPT = """You are a Code Executor Agent.

Your task:
1. Execute the file that was just fixed using execute_python_file tool
2. Report whether it ran successfully

If execution fails:
- Report the error clearly
- Identify what needs to be fixed

If execution succeeds:
- Confirm the code works correctly"""


REPORTER_PROMPT = """You are a Report Generator Agent.

Generate a final summary report including:
1. Project overview (files scanned)
2. Issues found
3. Fixes applied
4. Execution results
5. Final status (SUCCESS/NEEDS_MORE_WORK)

Be concise but informative. DO NOT call any tools."""


# ============================================================================
# AGENT NODES
# ============================================================================

def create_agent_executor(llm, tools: List, system_prompt: str, agent_name: str, max_turns: int = 10):
    """
    Create an agent that can have multi-turn conversations with tool use.
    Uses a ReAct-style loop: Think -> Act -> Observe -> (repeat)
    """
    has_tools = bool(tools)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    def agent_executor(state: WorkflowState) -> Dict[str, Any]:
        messages = list(state["messages"])
        step_logs = list(state.get("step_logs", []))
        
        # Add system prompt
        agent_messages = [SystemMessage(content=system_prompt)]
        
        # Add relevant context from history
        # If agent has no tools, filter out ToolMessages and AIMessages with tool_calls
        for msg in messages[-20:]:
            if not has_tools:
                # Skip ToolMessage for agents without tools
                if isinstance(msg, ToolMessage):
                    continue
                # Skip AIMessage with tool_calls
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Convert to plain text message
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
            
            # Log
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
                # No more tool calls - agent is done
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
# WORKFLOW NODES
# ============================================================================

class MultiAgentWorkflow:
    """
    Multi-agent workflow with proper feedback loops.
    
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
        
        workflow = StateGraph(WorkflowState)
        
        # Create agent executors
        scanner = create_agent_executor(
            self.fast_llm, [scan_project], SCANNER_PROMPT, "Scanner", max_turns=3
        )
        analyzer = create_agent_executor(
            self.llm, [check_syntax, read_file], ANALYZER_PROMPT, "Analyzer", max_turns=15
        )
        fixer = create_agent_executor(
            self.powerful_llm, [read_file, write_file], FIXER_PROMPT, "Fixer", max_turns=8
        )
        executor = create_agent_executor(
            self.fast_llm, [execute_python_file, run_pytest], EXECUTOR_PROMPT, "Executor", max_turns=3
        )
        reporter = create_agent_executor(
            self.llm, [], REPORTER_PROMPT, "Reporter", max_turns=2
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
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
            # Add initial message
            project_path = state["project_path"]
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=f"Scan the project at: {project_path}")
            ]
            
            result = scanner_fn(state)
            
            # Parse scan results from messages
            python_files = []
            test_files = []
            
            for msg in result.get("messages", []):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if "python_files" in content:
                    try:
                        # Find JSON in response
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
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
            # Add instruction
            files = state.get("python_files", [])
            project_path = state["project_path"]
            
            file_list = "\n".join(f"  - {f}" for f in files[:20])
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=f"Analyze these Python files in {project_path}:\n{file_list}\n\nCheck syntax of EACH file.")
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
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
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
                HumanMessage(content=f"Fix attempt {attempt}: Fix the file {current_file}\nError: {error_info}\n\nRead the file, fix the error, and write the corrected code.")
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
        def wrapper(state: WorkflowState) -> Dict[str, Any]:
            current_file = state.get("current_file")
            project_path = state["project_path"]
            
            # Build full path
            if current_file and not Path(current_file).is_absolute():
                full_path = str(Path(project_path) / current_file)
            else:
                full_path = current_file or project_path
            
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=f"Execute the fixed file: {full_path}\nVerify if the fix worked.")
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
    
    def _has_errors(self, state: WorkflowState) -> Literal["has_errors", "no_errors"]:
        """Check if there are files with errors"""
        errors = state.get("files_with_errors", [])
        
        if errors:
            print(f"\n[Decision] Found {len(errors)} file(s) with errors -> Go to Fixer")
            return "has_errors"
        else:
            print(f"\n[Decision] No errors found -> Go to Reporter")
            return "no_errors"
    
    def _execution_decision(self, state: WorkflowState) -> Literal["success", "retry", "give_up"]:
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
        """Run the workflow"""
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        project_path = str(Path(project_path).resolve())
        
        initial_state: WorkflowState = {
            "messages": [],
            "project_path": project_path,
            "project_name": Path(project_path).name,
            "python_files": [],
            "test_files": [],
            "files_with_errors": [],
            "current_file": None,
            "execution_results": [],
            "last_execution_success": False,
            "fix_attempts": 0,
            "max_fix_attempts": self.max_fix_attempts,
            "fixes_applied": [],
            "workflow_complete": False,
            "final_status": "",
            "step_logs": []
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*70}")
        print(f"  MULTI-AGENT CODE ANALYSIS WORKFLOW")
        print(f"{'='*70}")
        print(f"  Project: {project_path}")
        print(f"  Thread:  {thread_id}")
        print(f"  Max Fix Attempts: {self.max_fix_attempts}")
        print(f"{'='*70}\n")
        
        return self.graph.invoke(initial_state, config)
    
    def stream_run(self, project_path: str, thread_id: Optional[str] = None):
        """Stream the workflow execution"""
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        project_path = str(Path(project_path).resolve())
        
        initial_state: WorkflowState = {
            "messages": [],
            "project_path": project_path,
            "project_name": Path(project_path).name,
            "python_files": [],
            "test_files": [],
            "files_with_errors": [],
            "current_file": None,
            "execution_results": [],
            "last_execution_success": False,
            "fix_attempts": 0,
            "max_fix_attempts": self.max_fix_attempts,
            "fixes_applied": [],
            "workflow_complete": False,
            "final_status": "",
            "step_logs": []
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update


def create_multi_agent_workflow(llm_provider: str = "openrouter", max_fix_attempts: int = 5):
    """Factory function to create the workflow"""
    return MultiAgentWorkflow(llm_provider, max_fix_attempts)

