"""
LangGraph-based Multi-Agent System for Repository Analysis
Following LangGraph best practices from:
https://docs.langchain.com/oss/python/langgraph/workflows-agents

Key Features:
1. Proper Agent loop with tool calling cycle
2. Conditional edges for tool execution decisions
3. MessagesState for conversation management
4. Multi-turn conversation support
"""
import os
import sys
import ast
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
from datetime import datetime

from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from config import get_llm


# ============================================================================
# SHARED STATE - Using LangGraph's MessagesState pattern
# ============================================================================

class AgentState(TypedDict):
    """
    State for a single agent with tool calling loop.
    Following LangGraph's MessagesState pattern.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    agent_logs: List[Dict[str, Any]]


class RepoAnalysisState(TypedDict):
    """
    Shared state for the multi-agent workflow.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    project_path: str
    project_name: str
    python_files: List[str]
    test_files: List[str]
    files_to_analyze: List[str]
    current_file_index: int
    file_results: Dict[str, Dict[str, Any]]
    syntax_errors: List[Dict[str, Any]]
    test_results: List[Dict[str, Any]]
    errors_found: List[Dict[str, Any]]
    fixes_applied: List[Dict[str, Any]]
    phase: str
    should_continue: bool
    final_report: str
    agent_logs: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int


# ============================================================================
# TOOLS - Available to all agents
# ============================================================================

@tool
def scan_directory(path: str) -> str:
    """
    Scan a directory and return list of Python files.
    
    Args:
        path: Directory path to scan
        
    Returns:
        JSON string with python_files and test_files lists
    """
    root = Path(path)
    if not root.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})
    
    python_files = []
    test_files = []
    
    ignore_dirs = {'__pycache__', '.git', 'venv', '.venv', 'node_modules', 'build', 'dist'}
    
    for py_file in root.rglob("*.py"):
        if any(d in py_file.parts for d in ignore_dirs):
            continue
        
        rel_path = str(py_file.relative_to(root))
        
        if py_file.name.startswith("test_") or py_file.name.endswith("_test.py") or "tests" in py_file.parts:
            test_files.append(rel_path)
        else:
            python_files.append(rel_path)
    
    return json.dumps({
        "python_files": python_files,
        "test_files": test_files,
        "total": len(python_files) + len(test_files)
    })


@tool
def read_file(file_path: str) -> str:
    """
    Read content of a file.
    
    Args:
        file_path: Path to file to read
        
    Returns:
        File content or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        content = path.read_text(encoding='utf-8')
        return f"FILE: {file_path}\nLINES: {len(content.splitlines())}\n\n{content}"
    except Exception as e:
        return f"ERROR reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to file to write
        content: Content to write
        
    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"SUCCESS: Written {len(content)} chars to {file_path}"
    except Exception as e:
        return f"ERROR writing file: {str(e)}"


@tool
def check_syntax(file_path: str) -> str:
    """
    Check Python syntax of a file using AST.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        Syntax check result with any errors found
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"valid": False, "error": "File not found"})
        
        content = path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
            
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            
            return json.dumps({
                "valid": True,
                "file": file_path,
                "lines": len(content.splitlines()),
                "classes": classes,
                "functions": functions,
                "imports": imports
            })
            
        except SyntaxError as e:
            return json.dumps({
                "valid": False,
                "file": file_path,
                "error": f"Line {e.lineno}: {e.msg}",
                "line_number": e.lineno
            })
            
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


@tool
def run_python_file(file_path: str, timeout: int = 30) -> str:
    """
    Execute a Python file and capture output.
    
    Args:
        file_path: Path to Python file to run
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result with stdout, stderr, and exit code
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
            "stdout": result.stdout[:2000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else ""
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Execution timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def run_tests(project_path: str, test_framework: str = "pytest") -> str:
    """
    Run project tests using pytest or unittest.
    
    Args:
        project_path: Root path of the project
        test_framework: Either 'pytest' or 'unittest'
        
    Returns:
        Test results including passed/failed counts
    """
    try:
        if test_framework == "pytest":
            cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"]
        else:
            cmd = [sys.executable, "-m", "unittest", "discover", "-v"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_path
        )
        
        output = result.stdout + result.stderr
        
        passed = output.count(" PASSED") + output.count(" passed")
        failed = output.count(" FAILED") + output.count(" failed") + output.count(" ERROR")
        
        return json.dumps({
            "success": result.returncode == 0,
            "passed": passed,
            "failed": failed,
            "output": output[:3000]
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Tests timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def install_dependencies(project_path: str) -> str:
    """
    Install project dependencies from requirements.txt.
    
    Args:
        project_path: Root path of the project
        
    Returns:
        Installation result
    """
    req_file = Path(project_path) / "requirements.txt"
    
    if not req_file.exists():
        return json.dumps({"success": True, "message": "No requirements.txt found"})
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_path
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "message": "Dependencies installed" if result.returncode == 0 else result.stderr[:500]
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# All tools available to agents
ALL_REPO_TOOLS = [
    scan_directory,
    read_file,
    write_file,
    check_syntax,
    run_python_file,
    run_tests,
    install_dependencies
]


# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================

SCANNER_PROMPT = """You are a Project Scanner Agent. Your job is to:
1. Scan the project directory to find all Python files
2. Identify test files vs regular code files
3. Check for requirements.txt and other config files

Use the scan_directory tool to scan the project.
After scanning, summarize what you found.
When you have completed your task, respond with your findings WITHOUT calling more tools."""

ANALYZER_PROMPT = """You are a Code Analyzer Agent. Your job is to:
1. Analyze Python files for syntax errors and issues
2. Use the check_syntax tool to verify each file
3. Read files with read_file if you need more context
4. Identify problems that need fixing

Be thorough. Check multiple files if needed.
When you have completed your analysis, summarize the issues found WITHOUT calling more tools."""

TESTER_PROMPT = """You are a Test Runner Agent. Your job is to:
1. Run project tests using run_tests tool
2. Analyze test failures
3. Identify which files have failing tests

Report test results clearly with pass/fail counts.
When testing is complete, summarize results WITHOUT calling more tools."""

FIXER_PROMPT = """You are a Code Fixer Agent. Your job is to:
1. Fix syntax errors and bugs in Python files
2. Read the problematic file with read_file
3. Generate the corrected code
4. Write the fixed code with write_file

Make minimal changes to fix issues. Preserve the original intent.
After fixing, confirm what was changed WITHOUT calling more tools."""

REPORTER_PROMPT = """You are a Report Generator Agent. Your job is to:
1. Summarize all analysis and test results
2. List all issues found and fixes applied
3. Provide recommendations for improvement

Generate a clear, actionable report.
Do NOT call any tools - just summarize based on previous messages."""


# ============================================================================
# LANGGRAPH AGENT - Following best practices
# ============================================================================

class LangGraphAgent:
    """
    A proper LangGraph Agent with:
    1. Tool calling loop (llm_call -> should_continue -> tool_node -> llm_call)
    2. Conditional edges
    3. Multi-turn conversation support
    """
    
    def __init__(
        self, 
        llm,
        tools: List,
        system_prompt: str,
        agent_name: str = "Agent",
        max_iterations: int = 10
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.max_iterations = max_iterations
        
        # Bind tools to LLM
        self.llm_with_tools = llm.bind_tools(tools)
        
        # Create tool node using LangGraph's ToolNode
        self.tool_node = ToolNode(tools)
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build agent graph with proper tool calling loop"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("llm_call", self._llm_call)
        workflow.add_node("tool_node", self._tool_node)
        
        # Add edges
        workflow.add_edge(START, "llm_call")
        
        # Conditional edge: continue to tools or end
        workflow.add_conditional_edges(
            "llm_call",
            self._should_continue,
            {
                "tools": "tool_node",
                "end": END
            }
        )
        
        # After tools, go back to LLM
        workflow.add_edge("tool_node", "llm_call")
        
        return workflow.compile()
    
    def _llm_call(self, state: AgentState) -> Dict[str, Any]:
        """Call the LLM with current messages"""
        messages = state["messages"]
        
        # Ensure system prompt is first
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        
        # Call LLM
        response = self.llm_with_tools.invoke(messages)
        
        # Log response (handle encoding for Windows)
        content = response.content if hasattr(response, 'content') else str(response)
        # Remove emojis for Windows console compatibility
        safe_content = content.encode('ascii', 'ignore').decode('ascii')
        print(f"\n[{self.agent_name}] LLM Response:")
        print(f"  {safe_content[:300]}..." if len(safe_content) > 300 else f"  {safe_content}")
        
        if response.tool_calls:
            print(f"  Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        
        # Update agent logs
        agent_logs = state.get("agent_logs", [])
        agent_logs.append({
            "agent": self.agent_name,
            "type": "llm_call",
            "response": content,
            "tool_calls": [tc["name"] for tc in response.tool_calls] if response.tool_calls else [],
            "timestamp": datetime.now().isoformat()
        })
        
        return {"messages": [response], "agent_logs": agent_logs}
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tool calls from LLM"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not last_message.tool_calls:
            return {"messages": []}
        
        tool_results = []
        agent_logs = state.get("agent_logs", [])
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  [{self.agent_name}] Executing: {tool_name}")
            
            # Find and execute tool
            result = None
            for t in self.tools:
                if t.name == tool_name:
                    try:
                        result = t.invoke(tool_args)
                        safe_result = str(result).encode('ascii', 'ignore').decode('ascii')
                        print(f"    Result: {safe_result[:100]}...")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        print(f"    Error: {str(e)}")
                    break
            
            if result is None:
                result = f"Tool {tool_name} not found"
            
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
            
            agent_logs.append({
                "agent": self.agent_name,
                "type": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:200],
                "timestamp": datetime.now().isoformat()
            })
        
        return {"messages": tool_results, "agent_logs": agent_logs}
    
    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to continue calling tools or end"""
        messages = state["messages"]
        
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # Check iteration count to prevent infinite loops
        agent_logs = state.get("agent_logs", [])
        llm_calls = sum(1 for log in agent_logs if log.get("type") == "llm_call")
        
        if llm_calls >= self.max_iterations:
            print(f"  [{self.agent_name}] Max iterations ({self.max_iterations}) reached")
            return "end"
        
        # If LLM made tool calls, execute them
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, agent is done
        return "end"
    
    def run(self, initial_message: str, existing_messages: List = None) -> Dict[str, Any]:
        """Run the agent with a message"""
        messages = existing_messages or []
        messages = list(messages) + [HumanMessage(content=initial_message)]
        
        initial_state: AgentState = {
            "messages": messages,
            "agent_logs": []
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def stream(self, initial_message: str, existing_messages: List = None):
        """Stream agent execution"""
        messages = existing_messages or []
        messages = list(messages) + [HumanMessage(content=initial_message)]
        
        initial_state: AgentState = {
            "messages": messages,
            "agent_logs": []
        }
        
        for update in self.graph.stream(initial_state):
            yield update


# ============================================================================
# MULTI-AGENT WORKFLOW
# ============================================================================

class RepoAnalysisWorkflow:
    """
    Multi-agent workflow using proper LangGraph agents.
    Each agent has its own tool calling loop.
    """
    
    def __init__(self, llm_provider: str = "openrouter"):
        self.llm_provider = llm_provider
        
        # Create LLMs
        self.llm = get_llm(llm_provider, "default")
        self.fast_llm = get_llm(llm_provider, "fast")
        self.powerful_llm = get_llm(llm_provider, "powerful")
        
        # Create agents with proper tool calling loops
        self.scanner = LangGraphAgent(
            self.fast_llm, ALL_REPO_TOOLS, SCANNER_PROMPT, "Scanner", max_iterations=5
        )
        self.analyzer = LangGraphAgent(
            self.llm, ALL_REPO_TOOLS, ANALYZER_PROMPT, "Analyzer", max_iterations=8
        )
        self.tester = LangGraphAgent(
            self.fast_llm, ALL_REPO_TOOLS, TESTER_PROMPT, "Tester", max_iterations=5
        )
        self.fixer = LangGraphAgent(
            self.powerful_llm, ALL_REPO_TOOLS, FIXER_PROMPT, "Fixer", max_iterations=10
        )
        self.reporter = LangGraphAgent(
            self.llm, [], REPORTER_PROMPT, "Reporter", max_iterations=2
        )
        
        # Checkpointer for memory
        self.checkpointer = MemorySaver()
        
        # Build workflow
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow"""
        
        workflow = StateGraph(RepoAnalysisState)
        
        # Add agent nodes
        workflow.add_node("scanner", self._scanner_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("tester", self._tester_node)
        workflow.add_node("fixer", self._fixer_node)
        workflow.add_node("reporter", self._reporter_node)
        
        # Define flow
        workflow.add_edge(START, "scanner")
        workflow.add_edge("scanner", "analyzer")
        workflow.add_edge("analyzer", "tester")
        
        # Conditional: fix errors or report
        workflow.add_conditional_edges(
            "tester",
            self._should_fix,
            {
                "fix": "fixer",
                "report": "reporter"
            }
        )
        
        workflow.add_edge("fixer", "analyzer")  # Re-analyze after fix
        workflow.add_edge("reporter", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _scanner_node(self, state: RepoAnalysisState) -> Dict[str, Any]:
        """Scanner agent node"""
        print(f"\n{'='*60}")
        print(f"  SCANNER AGENT")
        print(f"{'='*60}")
        
        project_path = state["project_path"]
        result = self.scanner.run(f"Scan the project at: {project_path}")
        
        # Merge results
        return {
            "messages": result["messages"],
            "agent_logs": result["agent_logs"]
        }
    
    def _analyzer_node(self, state: RepoAnalysisState) -> Dict[str, Any]:
        """Analyzer agent node"""
        print(f"\n{'='*60}")
        print(f"  ANALYZER AGENT")
        print(f"{'='*60}")
        
        project_path = state["project_path"]
        existing_messages = state["messages"]
        
        result = self.analyzer.run(
            f"Analyze the Python files in {project_path}. Check syntax of each file.",
            existing_messages
        )
        
        return {
            "messages": result["messages"],
            "agent_logs": state.get("agent_logs", []) + result["agent_logs"]
        }
    
    def _tester_node(self, state: RepoAnalysisState) -> Dict[str, Any]:
        """Tester agent node"""
        print(f"\n{'='*60}")
        print(f"  TESTER AGENT")
        print(f"{'='*60}")
        
        project_path = state["project_path"]
        existing_messages = state["messages"]
        
        result = self.tester.run(
            f"Run tests in {project_path} and report results.",
            existing_messages
        )
        
        return {
            "messages": result["messages"],
            "agent_logs": state.get("agent_logs", []) + result["agent_logs"]
        }
    
    def _fixer_node(self, state: RepoAnalysisState) -> Dict[str, Any]:
        """Fixer agent node"""
        print(f"\n{'='*60}")
        print(f"  FIXER AGENT")
        print(f"{'='*60}")
        
        iteration = state.get("iteration_count", 0) + 1
        max_iter = state.get("max_iterations", 3)
        
        existing_messages = state["messages"]
        
        result = self.fixer.run(
            "Fix the errors found in the previous analysis. Read the file, fix the issues, and write the corrected code.",
            existing_messages
        )
        
        return {
            "messages": result["messages"],
            "agent_logs": state.get("agent_logs", []) + result["agent_logs"],
            "iteration_count": iteration
        }
    
    def _reporter_node(self, state: RepoAnalysisState) -> Dict[str, Any]:
        """Reporter agent node"""
        print(f"\n{'='*60}")
        print(f"  REPORTER AGENT")
        print(f"{'='*60}")
        
        existing_messages = state["messages"]
        
        result = self.reporter.run(
            "Generate a final report summarizing all findings and fixes.",
            existing_messages
        )
        
        return {
            "messages": result["messages"],
            "agent_logs": state.get("agent_logs", []) + result["agent_logs"]
        }
    
    def _should_fix(self, state: RepoAnalysisState) -> Literal["fix", "report"]:
        """Decide whether to fix errors or generate report"""
        
        # Check iteration count
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 3)
        
        if iteration >= max_iter:
            print(f"  Max fix iterations ({max_iter}) reached")
            return "report"
        
        # Check messages for errors
        messages = state.get("messages", [])
        
        for msg in reversed(messages[-5:]):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            content_lower = content.lower()
            
            # Check for syntax errors
            if '"valid": false' in content_lower or '"valid":false' in content_lower:
                print("  Syntax errors found - going to fixer")
                return "fix"
            
            # Check for test failures
            if '"failed":' in content and '"failed": 0' not in content and '"failed":0' not in content:
                print("  Test failures found - going to fixer")
                return "fix"
        
        print("  No critical errors - going to reporter")
        return "report"
    
    def run(self, project_path: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the workflow"""
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        project_path = str(Path(project_path).resolve())
        
        initial_state: RepoAnalysisState = {
            "messages": [HumanMessage(content=f"Analyze project: {project_path}")],
            "project_path": project_path,
            "project_name": Path(project_path).name,
            "python_files": [],
            "test_files": [],
            "files_to_analyze": [],
            "current_file_index": 0,
            "file_results": {},
            "syntax_errors": [],
            "test_results": [],
            "errors_found": [],
            "fixes_applied": [],
            "phase": "scan",
            "should_continue": True,
            "final_report": "",
            "agent_logs": [],
            "iteration_count": 0,
            "max_iterations": 3
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*70}")
        print(f"  REPO ANALYSIS WORKFLOW - LangGraph Multi-Agent System")
        print(f"{'='*70}")
        print(f"  Project: {project_path}")
        print(f"  Thread:  {thread_id}")
        print(f"{'='*70}\n")
        
        return self.graph.invoke(initial_state, config)
    
    def stream_run(self, project_path: str, thread_id: Optional[str] = None):
        """Stream the workflow execution"""
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        project_path = str(Path(project_path).resolve())
        
        initial_state: RepoAnalysisState = {
            "messages": [HumanMessage(content=f"Analyze project: {project_path}")],
            "project_path": project_path,
            "project_name": Path(project_path).name,
            "python_files": [],
            "test_files": [],
            "files_to_analyze": [],
            "current_file_index": 0,
            "file_results": {},
            "syntax_errors": [],
            "test_results": [],
            "errors_found": [],
            "fixes_applied": [],
            "phase": "scan",
            "should_continue": True,
            "final_report": "",
            "agent_logs": [],
            "iteration_count": 0,
            "max_iterations": 3
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_repo_workflow(llm_provider: str = "openrouter") -> RepoAnalysisWorkflow:
    """Create a new repository analysis workflow"""
    return RepoAnalysisWorkflow(llm_provider)
