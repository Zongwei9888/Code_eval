#!/usr/bin/env python
"""
CODE EVAL CLI v3.0
Command-line interface for Autonomous Multi-Agent Code Analysis System

Features:
1. Quick Scan - Fast local scan (no AI)
2. Single File Analysis - Autonomous single-file agent (Analyzer -> Fixer -> Executor)
3. Multi-Agent Analysis - Full autonomous system with 4 core agents

AUTONOMOUS AGENT SYSTEM:
- Supervisor LLM observes state and decides each step dynamically
- 4 Core Agents: scanner, analyzer, fixer, executor
- Each agent has dedicated tools for their specific role
- Dynamic routing based on error analysis

Key Innovation:
- OLD: Hardcoded flow (Scanner -> Analyzer -> Fixer -> ...)
- NEW: Supervisor decides dynamically based on LLM reasoning
"""
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# TERMINAL COLORS (No external dependencies)
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-TTY outputs"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_terminal_width() -> int:
    """Get terminal width"""
    try:
        return os.get_terminal_size().columns
    except:
        return 80


def print_header(title: str, subtitle: str = ""):
    """Print styled header"""
    width = get_terminal_width()
    
    print(f"\n{Colors.BRIGHT_CYAN}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}  {title}{Colors.RESET}")
    if subtitle:
        print(f"{Colors.DIM}  {subtitle}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{'=' * width}{Colors.RESET}\n")


def print_section(title: str):
    """Print section divider"""
    width = get_terminal_width()
    print(f"\n{Colors.CYAN}{'-' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * width}{Colors.RESET}")


def print_status(status: str, message: str):
    """Print status message with color"""
    status_colors = {
        "ok": Colors.BRIGHT_GREEN,
        "info": Colors.BRIGHT_BLUE,
        "warn": Colors.BRIGHT_YELLOW,
        "error": Colors.BRIGHT_RED,
        "run": Colors.BRIGHT_MAGENTA
    }
    color = status_colors.get(status, Colors.WHITE)
    symbol = {"ok": "[+]", "info": "[*]", "warn": "[!]", "error": "[X]", "run": "[~]"}.get(status, "[-]")
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.DIM}{timestamp}{Colors.RESET} {color}{symbol}{Colors.RESET} {message}")


def print_box(title: str, content: List[str], color: str = Colors.CYAN):
    """Print content in a box"""
    width = min(get_terminal_width() - 4, 70)
    
    print(f"\n{color}+{'-' * (width - 2)}+{Colors.RESET}")
    print(f"{color}|{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET}{' ' * (width - len(title) - 4)}{color}|{Colors.RESET}")
    print(f"{color}+{'-' * (width - 2)}+{Colors.RESET}")
    
    for line in content:
        line = line[:width - 4]
        padding = width - len(line) - 4
        print(f"{color}|{Colors.RESET} {line}{' ' * padding} {color}|{Colors.RESET}")
    
    print(f"{color}+{'-' * (width - 2)}+{Colors.RESET}\n")


# ============================================================================
# CHAT BOX DISPLAY (For LLM Responses)
# ============================================================================

def print_chat_message(role: str, content: str, turn: int = 0, has_tools: bool = False):
    """Print a chat message in a styled box"""
    width = min(get_terminal_width() - 6, 75)
    
    # Role colors and styles - Updated for 10 autonomous agents
    role_styles = {
        "llm": (Colors.BRIGHT_MAGENTA, "LLM"),
        "tool": (Colors.BRIGHT_BLUE, "TOOL"),
        "system": (Colors.BRIGHT_CYAN, "SYS"),
        "user": (Colors.BRIGHT_GREEN, "USER"),
        "error": (Colors.BRIGHT_RED, "ERR"),
        # 10 Autonomous Agents
        "supervisor": (Colors.BRIGHT_MAGENTA, "SUPERVISOR"),
        "planner": (Colors.BRIGHT_BLUE, "PLANNER"),
        "researcher": (Colors.MAGENTA, "RESEARCHER"),
        "scanner": (Colors.BRIGHT_CYAN, "SCANNER"),
        "analyzer": (Colors.BRIGHT_YELLOW, "ANALYZER"),
        "fixer": (Colors.BRIGHT_GREEN, "FIXER"),
        "executor": (Colors.BRIGHT_BLUE, "EXECUTOR"),
        "tester": (Colors.YELLOW, "TESTER"),
        "reviewer": (Colors.BRIGHT_MAGENTA, "REVIEWER"),
        "environment": (Colors.GREEN, "ENVIRONMENT"),
        "git": (Colors.CYAN, "GIT"),
        "reporter": (Colors.BRIGHT_MAGENTA, "REPORTER"),  # Legacy
    }
    
    color, label = role_styles.get(role.lower(), (Colors.WHITE, role.upper()))
    
    # Header
    turn_str = f" T{turn}" if turn > 0 else ""
    tool_indicator = f" {Colors.YELLOW}>> tools{Colors.RESET}" if has_tools else ""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print()
    print(f"  {color}{'_' * (width - 2)}{Colors.RESET}")
    print(f"  {color}|{Colors.RESET} {Colors.BOLD}{label}{turn_str}{Colors.RESET} {Colors.DIM}{timestamp}{Colors.RESET}{tool_indicator}")
    print(f"  {color}|{'-' * (width - 3)}{Colors.RESET}")
    
    # Content - word wrap
    lines = []
    for paragraph in content.split('\n'):
        if not paragraph.strip():
            lines.append("")
            continue
        
        words = paragraph.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width - 6:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    
    # Print content lines (limit to 8 lines for readability)
    for i, line in enumerate(lines[:8]):
        print(f"  {color}|{Colors.RESET}  {line}")
    
    if len(lines) > 8:
        print(f"  {color}|{Colors.RESET}  {Colors.DIM}... ({len(lines) - 8} more lines){Colors.RESET}")
    
    print(f"  {color}|{'_' * (width - 3)}{Colors.RESET}")


def print_tool_call(tool_name: str, result: str, success: bool = True):
    """Print a tool call result in a compact box"""
    width = min(get_terminal_width() - 8, 70)
    
    color = Colors.BRIGHT_GREEN if success else Colors.BRIGHT_RED
    status = "OK" if success else "FAIL"
    
    print(f"    {Colors.DIM}+{'-' * (width - 4)}+{Colors.RESET}")
    print(f"    {Colors.DIM}|{Colors.RESET} {Colors.BRIGHT_BLUE}[TOOL]{Colors.RESET} {tool_name} {color}[{status}]{Colors.RESET}")
    print(f"    {Colors.DIM}|{'-' * (width - 5)}{Colors.RESET}")
    
    # Truncate result
    result_clean = result.replace('\n', ' ')[:width - 10]
    print(f"    {Colors.DIM}|{Colors.RESET}  {Colors.DIM}{result_clean}...{Colors.RESET}")
    print(f"    {Colors.DIM}+{'-' * (width - 4)}+{Colors.RESET}")


def print_agent_header(agent_name: str, step: int):
    """Print a prominent agent header"""
    width = get_terminal_width()
    
    agent_colors = {
        "supervisor": Colors.BRIGHT_MAGENTA,
        "planner": Colors.BRIGHT_BLUE,
        "researcher": Colors.MAGENTA,
        "scanner": Colors.BRIGHT_CYAN,
        "analyzer": Colors.BRIGHT_YELLOW,
        "fixer": Colors.BRIGHT_GREEN,
        "executor": Colors.BRIGHT_BLUE,
        "tester": Colors.YELLOW,
        "reviewer": Colors.BRIGHT_MAGENTA,
        "environment": Colors.GREEN,
        "git": Colors.CYAN,
        "reporter": Colors.BRIGHT_MAGENTA,  # Legacy
    }
    
    color = agent_colors.get(agent_name.lower(), Colors.BRIGHT_CYAN)
    
    print()
    print(f"{color}{'=' * width}{Colors.RESET}")
    print(f"{color}  STEP {step} | {agent_name.upper()} AGENT{Colors.RESET}")
    print(f"{color}{'=' * width}{Colors.RESET}")


def print_table(headers: List[str], rows: List[List[str]]):
    """Print formatted table"""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"\n{Colors.BOLD}{Colors.CYAN}{header_line}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * len(header_line)}{Colors.RESET}")
    
    # Print rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(row_line)
    print()


def format_time():
    """Get formatted timestamp"""
    return datetime.now().strftime("%H:%M:%S")


# ============================================================================
# PROJECT MANAGEMENT
# ============================================================================

BENCH_DIR = Path(__file__).parent / "code_bench"


def get_projects() -> List[str]:
    """Get available projects in code_bench"""
    BENCH_DIR.mkdir(exist_ok=True)
    projects = []
    for d in BENCH_DIR.iterdir():
        if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__':
            projects.append(d.name)
    return sorted(projects)


def get_files_in_project(project_name: str) -> List[str]:
    """Get Python files in a project"""
    if not project_name:
        return []
    path = BENCH_DIR / project_name
    if not path.exists():
        return []
    
    files = []
    for f in path.rglob("*.py"):
        if "__pycache__" not in str(f) and not f.name.startswith('.'):
            files.append(str(f.relative_to(path)))
    return sorted(files)


def select_project() -> Optional[str]:
    """Interactive project selection"""
    projects = get_projects()
    
    if not projects:
        print_status("warn", "No projects found in code_bench/")
        print_status("info", "Place your Python projects in the code_bench/ directory")
        return None
    
    print(f"\n{Colors.BOLD}Available Projects:{Colors.RESET}\n")
    
    for i, project in enumerate(projects, 1):
        path = BENCH_DIR / project
        py_files = list(path.rglob("*.py"))
        py_count = len([f for f in py_files if "__pycache__" not in str(f)])
        print(f"  {Colors.BRIGHT_CYAN}{i:2}{Colors.RESET}. {project} {Colors.DIM}({py_count} Python files){Colors.RESET}")
    
    print(f"\n  {Colors.DIM} 0. Cancel{Colors.RESET}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BRIGHT_YELLOW}Select project [1-{len(projects)}]: {Colors.RESET}").strip()
            if choice == '0' or choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(projects):
                return projects[idx]
            print_status("error", "Invalid selection")
        except ValueError:
            print_status("error", "Please enter a number")
        except KeyboardInterrupt:
            return None


def select_file(project_name: str) -> Optional[str]:
    """Interactive file selection within a project"""
    files = get_files_in_project(project_name)
    
    if not files:
        print_status("warn", f"No Python files found in {project_name}")
        return None
    
    print(f"\n{Colors.BOLD}Python Files in {project_name}:{Colors.RESET}\n")
    
    for i, file in enumerate(files, 1):
        print(f"  {Colors.BRIGHT_CYAN}{i:2}{Colors.RESET}. {file}")
    
    print(f"\n  {Colors.DIM} 0. Cancel{Colors.RESET}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BRIGHT_YELLOW}Select file [1-{len(files)}]: {Colors.RESET}").strip()
            if choice == '0' or choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            print_status("error", "Invalid selection")
        except ValueError:
            print_status("error", "Please enter a number")
        except KeyboardInterrupt:
            return None


# ============================================================================
# QUICK SCAN (No AI)
# ============================================================================

def run_quick_scan(project_name: str):
    """Run quick local scan without AI"""
    from agent.repo_agents import quick_scan
    
    project_path = BENCH_DIR / project_name
    
    print_header("QUICK SCAN", f"Project: {project_name}")
    print_status("run", "Scanning project structure...")
    
    try:
        result = quick_scan(str(project_path))
        
        # Summary
        print_section("Scan Results")
        
        has_errors = result["files_with_syntax_errors"] > 0
        
        summary = [
            f"Total Python files: {result['total_files']}",
            f"Syntax errors: {result['files_with_syntax_errors']}",
            f"Has tests: {'Yes' if result['has_tests'] else 'No'}"
        ]
        
        color = Colors.RED if has_errors else Colors.GREEN
        print_box(result["project_name"], summary, color)
        
        # Syntax errors details
        if result["syntax_errors"]:
            print_section("Syntax Errors")
            for err in result["syntax_errors"]:
                print(f"\n  {Colors.BRIGHT_RED}{err['file']}{Colors.RESET}")
                for e in err["errors"][:3]:
                    print(f"    {Colors.DIM}- {e}{Colors.RESET}")
        else:
            print_status("ok", "All files have valid syntax!")
        
        # File tree
        print_section("Project Structure")
        print_file_tree(project_path)
        
    except Exception as e:
        print_status("error", f"Scan failed: {str(e)}")


def print_file_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """Print directory tree"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(items[:30]):
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            
            is_last = i == len(items) - 1
            connector = "+-" if is_last else "|-"
            
            if item.is_dir():
                print(f"  {prefix}{connector} {Colors.BRIGHT_BLUE}{item.name}/{Colors.RESET}")
                new_prefix = prefix + ("   " if is_last else "|  ")
                print_file_tree(item, new_prefix, max_depth, current_depth + 1)
            else:
                color = Colors.BRIGHT_GREEN if item.suffix == ".py" else Colors.DIM
                print(f"  {prefix}{connector} {color}{item.name}{Colors.RESET}")
    except:
        pass


# ============================================================================
# SINGLE FILE ANALYSIS
# ============================================================================

def run_single_file_analysis(project_name: str, file_path: str, provider: str = "openrouter", auto_fix: bool = True):
    """Run autonomous single-file analysis with Supervisor decision-making"""
    from workflow.code_workflow import create_workflow
    
    full_path = BENCH_DIR / project_name / file_path
    
    if not full_path.exists():
        print_status("error", f"File not found: {file_path}")
        return
    
    print_header("AUTONOMOUS SINGLE-FILE AGENT", f"File: {file_path}")
    
    print_status("info", f"Provider: {provider}")
    print_status("info", f"Auto-fix: {auto_fix}")
    print_status("info", f"Max attempts: {5 if auto_fix else 1}")
    print_status("ok", "Mode: Supervisor decides (Analyzer → Fixer → Executor)")
    
    try:
        # Create workflow
        print_status("run", "Creating workflow...")
        workflow = create_workflow(
            llm_provider=provider,
            max_attempts=5 if auto_fix else 1
        )
        
        # Read file
        original_code = full_path.read_text(encoding='utf-8')
        print_status("ok", f"Loaded file: {len(original_code)} characters")
        
        # Run workflow
        print_section("Workflow Execution")
        thread_id = str(uuid.uuid4())
        print_status("info", f"Thread: {thread_id[:8]}...")
        
        step = 0
        current_node = None
        for update in workflow.stream_run(str(full_path), original_code, thread_id=thread_id):
            step += 1
            if update:
                node_name = list(update.keys())[0]
                node_data = update[node_name]
                
                # Print agent header when node changes
                if node_name != current_node:
                    print_agent_header(node_name.replace("_node", ""), step)
                    current_node = node_name
                
                if isinstance(node_data, dict):
                    # Show analysis in chat box
                    if node_data.get("code_analysis"):
                        analysis = node_data["code_analysis"]
                        print_chat_message("analyzer", analysis[:600], turn=step)
                    
                    # Show execution result
                    if "execution_success" in node_data:
                        success = node_data["execution_success"]
                        if success:
                            print_chat_message("executor", "Code executed successfully! No errors detected.", turn=step)
                        else:
                            error_msg = node_data.get("last_error", "Unknown error")
                            print_chat_message("error", f"Execution failed:\n{error_msg[:400]}", turn=step)
                    
                    # Show modifications
                    if node_data.get("modification_history"):
                        mods = node_data["modification_history"]
                        if isinstance(mods, list) and mods:
                            last_mod = mods[-1] if isinstance(mods[-1], str) else str(mods[-1])
                            print_chat_message("fixer", f"Applied code modification:\n{last_mod[:300]}", turn=step)
        
        # Final results
        print_section("Results")
        
        # Check if file was modified
        try:
            final_code = full_path.read_text(encoding='utf-8')
            modified = final_code != original_code
            
            orig_lines = len(original_code.splitlines())
            final_lines = len(final_code.splitlines())
            
            summary = [
                f"Steps: {step}",
                f"Modified: {'Yes' if modified else 'No'}",
                f"Lines: {orig_lines} -> {final_lines}",
                f"Chars: {len(original_code)} -> {len(final_code)}"
            ]
            
            color = Colors.BRIGHT_GREEN if modified else Colors.BRIGHT_YELLOW
            print_box("FILE STATUS", summary, color)
            
        except Exception as e:
            print_status("warn", f"Could not read final file: {e}")
        
        print_status("ok", f"Completed in {step} steps")
        
    except Exception as e:
        print_status("error", f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MULTI-AGENT ANALYSIS
# ============================================================================

def run_multi_agent_analysis(project_name: str, provider: str = "openrouter", max_attempts: int = 5):
    """Run autonomous multi-agent analysis with 10 specialist agents"""
    from workflow import create_multi_agent_workflow
    
    project_path = BENCH_DIR / project_name
    
    print_header("AUTONOMOUS MULTI-AGENT SYSTEM", f"Project: {project_name}")
    
    print_status("info", f"Provider: {provider}")
    print_status("info", f"Max iterations: {max_attempts}")
    print_status("ok", "Mode: AUTONOMOUS (Supervisor LLM decides each step)")
    print_status("info", "4 Core Agents: scanner, analyzer, fixer, executor")
    
    try:
        # Create workflow
        print_status("run", "Creating multi-agent workflow...")
        workflow = create_multi_agent_workflow(provider, max_fix_attempts=max_attempts)
        
        # Run workflow
        print_section("Workflow Execution")
        
        step_count = 0
        all_step_logs = []
        final_state = None
        current_agent = None
        
        for update in workflow.stream_run(str(project_path)):
            step_count += 1
            if update:
                node = list(update.keys())[0]
                node_data = update.get(node, {})
                
                # Print agent header when agent changes
                if node != current_agent:
                    print_agent_header(node, step_count)
                    current_agent = node
                
                if isinstance(node_data, dict):
                    # Process step logs
                    step_logs = node_data.get("step_logs", [])
                    new_logs = step_logs[len(all_step_logs):]
                    all_step_logs = step_logs
                    
                    for sl in new_logs:
                        agent = sl.get("agent", "")
                        turn = sl.get("turn", 0)
                        log_type = sl.get("type", "")
                        
                        if log_type == "llm_response":
                            content = sl.get("content", "")
                            has_tools = sl.get("has_tool_calls", False)
                            # Use chat box for LLM responses
                            print_chat_message(agent, content[:500], turn, has_tools)
                        elif log_type == "tool_call":
                            tool = sl.get("tool", "unknown")
                            result = sl.get("result", "")
                            success = "error" not in result.lower() and "fail" not in result.lower()
                            print_tool_call(tool, result[:200], success)
                    
                    # Show key state changes in a status box
                    state_changes = []
                    if node_data.get("files_with_errors"):
                        errs = node_data["files_with_errors"]
                        state_changes.append(f"Found {len(errs)} file(s) with errors")
                    
                    if "fix_attempts" in node_data and node_data["fix_attempts"] > 0:
                        state_changes.append(f"Fix attempt #{node_data['fix_attempts']}")
                    
                    if "last_execution_success" in node_data:
                        success = node_data["last_execution_success"]
                        state_changes.append(f"Execution: {'SUCCESS' if success else 'FAILED'}")
                    
                    if state_changes:
                        print()
                        for change in state_changes:
                            if "SUCCESS" in change:
                                print_status("ok", change)
                            elif "FAILED" in change or "error" in change.lower():
                                print_status("error", change)
                            else:
                                print_status("info", change)
                
                final_state = node_data
            
            if step_count > 30:
                print_status("warn", "Reached iteration limit")
                break
        
        # Final summary
        print_section("Summary")
        
        # Calculate stats
        agents_seen = set()
        llm_calls = 0
        tool_calls = 0
        for sl in all_step_logs:
            agents_seen.add(sl.get("agent", ""))
            if sl.get("type") == "llm_response":
                llm_calls += 1
            elif sl.get("type") == "tool_call":
                tool_calls += 1
        
        exec_success = final_state.get("last_execution_success", False) if final_state else False
        fix_attempts = final_state.get("fix_attempts", 0) if final_state else 0
        
        # Print summary in a styled box
        summary_content = [
            f"Total Steps: {step_count}",
            f"Agents Used: {', '.join(sorted(agents_seen))}",
            f"LLM Calls: {llm_calls}",
            f"Tool Calls: {tool_calls}",
            f"Fix Attempts: {fix_attempts}",
            f"Final Status: {'SUCCESS' if exec_success else 'INCOMPLETE'}"
        ]
        
        color = Colors.BRIGHT_GREEN if exec_success else Colors.BRIGHT_YELLOW
        print_box("ANALYSIS SUMMARY", summary_content, color)
        
        if exec_success:
            print_status("ok", "Analysis completed successfully!")
        else:
            print_status("warn", f"Analysis incomplete after {fix_attempts} fix attempts")
        
    except Exception as e:
        print_status("error", f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TRUE AGENT ANALYSIS (Autonomous Decision-Making)
# ============================================================================

def run_true_agent_analysis(project_name: str, provider: str = "openrouter", max_iterations: int = 20):
    """
    Run Autonomous Agent System with 10 specialists
    
    The Autonomous Agent uses a Supervisor LLM to:
    - Observe current state comprehensively
    - Reason about what to do next using LLM intelligence
    - Dynamically select from 10 specialist agents
    - Adapt strategy based on feedback loops
    - Each agent has dedicated tools for their role
    
    This is the SAME as run_multi_agent_analysis (unified autonomous system)
    """
    from workflow import create_true_agent
    
    project_path = BENCH_DIR / project_name
    
    print_header("AUTONOMOUS AGENT SYSTEM", f"Project: {project_name}")
    
    print_status("info", f"Provider: {provider}")
    print_status("info", f"Max iterations: {max_iterations}")
    print_status("ok", "Mode: AUTONOMOUS (Supervisor decides dynamically)")
    
    print()
    print(f"  {Colors.BRIGHT_MAGENTA}Autonomous Decision-Making:{Colors.RESET}")
    print(f"  {Colors.DIM}  - 4 Core Agents: scanner, analyzer, fixer, executor{Colors.RESET}")
    print(f"  {Colors.DIM}  - Supervisor LLM observes state and decides each step{Colors.RESET}")
    print(f"  {Colors.DIM}  - Dynamic routing based on error analysis{Colors.RESET}")
    print()
    
    try:
        # Create True Agent
        print_status("run", "Creating True Agent...")
        agent = create_true_agent(
            llm_provider=provider,
            max_iterations=max_iterations
        )
        
        print_status("ok", f"Specialists: {', '.join(agent.specialists.keys())}")
        
        # Get user request
        print()
        default_request = "Analyze the project and fix any code issues"
        user_request = input(
            f"{Colors.BRIGHT_YELLOW}Enter request [{default_request}]: {Colors.RESET}"
        ).strip() or default_request
        
        # Run agent
        print_section("Agent Execution")
        
        thread_id = str(uuid.uuid4())
        print_status("info", f"Thread: {thread_id[:8]}...")
        print_status("info", f"Request: {user_request}")
        print()
        
        step_count = 0
        all_decisions = []
        final_state = None
        current_agent = None
        
        for update in agent.stream_run(str(project_path), user_request, thread_id):
            step_count += 1
            if update:
                node = list(update.keys())[0]
                node_data = update.get(node, {})
                
                # Print agent header when agent changes
                if node != current_agent:
                    if node == "supervisor":
                        print(f"\n{Colors.BRIGHT_MAGENTA}{'='*60}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_MAGENTA}  SUPERVISOR - Making Decision{Colors.RESET}")
                        print(f"{Colors.BRIGHT_MAGENTA}{'='*60}{Colors.RESET}")
                    else:
                        print_agent_header(node, step_count)
                    current_agent = node
                
                if isinstance(node_data, dict):
                    # Show supervisor decision
                    if node == "supervisor":
                        decision = node_data.get("supervisor_decision", "")
                        reasoning = node_data.get("supervisor_reasoning", "")
                        iteration = node_data.get("iteration_count", 0)
                        
                        if decision:
                            print()
                            print(f"  {Colors.BOLD}Iteration:{Colors.RESET} {iteration}")
                            print(f"  {Colors.BOLD}Decision:{Colors.RESET} {Colors.BRIGHT_CYAN}{decision}{Colors.RESET}")
                            if reasoning:
                                # Truncate reasoning for display
                                reasoning_short = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                                print(f"  {Colors.BOLD}Reasoning:{Colors.RESET} {Colors.DIM}{reasoning_short}{Colors.RESET}")
                            
                            all_decisions.append({
                                "iteration": iteration,
                                "decision": decision,
                                "reasoning": reasoning[:100]
                            })
                    
                    # Show step logs
                    step_logs = node_data.get("step_logs", [])
                    if step_logs:
                        latest_log = step_logs[-1]
                        agent_name = latest_log.get("agent", "")
                        action = latest_log.get("action", "")
                        if action and agent_name != "supervisor":
                            print_status("info", f"[{agent_name}] {action}")
                    
                    # Show key state changes
                    if node_data.get("syntax_errors"):
                        errors = node_data["syntax_errors"]
                        print_status("warn", f"Syntax errors found: {len(errors)}")
                    
                    if node_data.get("last_execution_success") is not None:
                        success = node_data["last_execution_success"]
                        if success:
                            print_status("ok", "Execution succeeded!")
                        else:
                            error_msg = node_data.get("last_error_message", "")[:100]
                            print_status("error", f"Execution failed: {error_msg}")
                    
                    if node_data.get("goal_achieved"):
                        print_status("ok", "Goal achieved!")
                
                final_state = node_data
            
            if step_count > 50:
                print_status("warn", "Reached step limit")
                break
        
        # Final summary
        print_section("Summary")
        
        if final_state:
            iterations = final_state.get("iteration_count", 0)
            modifications = len(final_state.get("modifications", []))
            goal_achieved = final_state.get("goal_achieved", False)
            
            summary_content = [
                f"Total Steps: {step_count}",
                f"Iterations: {iterations}",
                f"Decisions Made: {len(all_decisions)}",
                f"Modifications: {modifications}",
                f"Goal Achieved: {'YES' if goal_achieved else 'NO'}"
            ]
            
            # Show decision history
            if all_decisions:
                summary_content.append("")
                summary_content.append("Decision History:")
                for d in all_decisions[-5:]:
                    summary_content.append(f"  {d['iteration']}: {d['decision']}")
            
            color = Colors.BRIGHT_GREEN if goal_achieved else Colors.BRIGHT_YELLOW
            print_box("TRUE AGENT RESULTS", summary_content, color)
            
            if goal_achieved:
                print_status("ok", "True Agent completed successfully!")
            else:
                print_status("warn", f"Agent stopped after {iterations} iterations")
        else:
            print_status("warn", "No final state available")
        
    except Exception as e:
        print_status("error", f"True Agent failed: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_main_menu():
    """Show main interactive menu"""
    clear_screen()
    
    print(f"""
{Colors.BRIGHT_CYAN}
   ██████╗ ██████╗ ██████╗ ███████╗    ███████╗██╗   ██╗ █████╗ ██╗     
  ██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔════╝██║   ██║██╔══██╗██║     
  ██║     ██║   ██║██║  ██║█████╗      █████╗  ██║   ██║███████║██║     
  ██║     ██║   ██║██║  ██║██╔══╝      ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     
  ╚██████╗╚██████╔╝██████╔╝███████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗
   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
{Colors.RESET}
{Colors.DIM}   Autonomous Multi-Agent Code Analysis System | CLI v3.0{Colors.RESET}
""")
    
    print(f"{Colors.BOLD}  Main Menu{Colors.RESET}\n")
    print(f"  {Colors.BRIGHT_CYAN}1{Colors.RESET}. Quick Scan        {Colors.DIM}- Fast local scan (no AI){Colors.RESET}")
    print(f"  {Colors.BRIGHT_CYAN}2{Colors.RESET}. Single File Fix   {Colors.DIM}- Autonomous single-file agent{Colors.RESET}")
    print(f"  {Colors.BRIGHT_MAGENTA}3{Colors.RESET}. {Colors.BRIGHT_MAGENTA}Multi-Agent{Colors.RESET}        {Colors.DIM}- 4 core agents + Supervisor{Colors.RESET}")
    print(f"  {Colors.BRIGHT_MAGENTA}4{Colors.RESET}. {Colors.BRIGHT_MAGENTA}Autonomous Agent{Colors.RESET}   {Colors.DIM}- Same as #3 (unified system){Colors.RESET}")
    print(f"  {Colors.BRIGHT_CYAN}5{Colors.RESET}. List Projects     {Colors.DIM}- Show available projects{Colors.RESET}")
    print(f"  {Colors.BRIGHT_CYAN}6{Colors.RESET}. Settings          {Colors.DIM}- Configure LLM provider{Colors.RESET}")
    print()
    print(f"  {Colors.DIM}q. Quit{Colors.RESET}")
    print()


def settings_menu(current_provider: str) -> str:
    """Settings menu for configuration"""
    from config import MODEL_MAPPINGS
    
    print_section("Settings")
    
    providers = list(MODEL_MAPPINGS.keys())
    print(f"\n{Colors.BOLD}LLM Providers:{Colors.RESET}\n")
    
    for i, provider in enumerate(providers, 1):
        current = " (current)" if provider == current_provider else ""
        print(f"  {Colors.BRIGHT_CYAN}{i}{Colors.RESET}. {provider}{Colors.DIM}{current}{Colors.RESET}")
    
    print(f"\n  {Colors.DIM}0. Cancel{Colors.RESET}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BRIGHT_YELLOW}Select provider [1-{len(providers)}]: {Colors.RESET}").strip()
            if choice == '0':
                return current_provider
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                new_provider = providers[idx]
                print_status("ok", f"Provider set to: {new_provider}")
                return new_provider
            print_status("error", "Invalid selection")
        except ValueError:
            print_status("error", "Please enter a number")
        except KeyboardInterrupt:
            return current_provider


def interactive_mode():
    """Run in interactive mode"""
    from config import DEFAULT_PROVIDER
    
    current_provider = DEFAULT_PROVIDER
    
    while True:
        show_main_menu()
        
        try:
            choice = input(f"{Colors.BRIGHT_YELLOW}Select option [1-6]: {Colors.RESET}").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}\n")
                break
            
            elif choice == '1':
                # Quick Scan
                project = select_project()
                if project:
                    run_quick_scan(project)
                    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
            elif choice == '2':
                # Single File Fix
                project = select_project()
                if project:
                    file = select_file(project)
                    if file:
                        auto_fix = input(f"\n{Colors.BRIGHT_YELLOW}Enable auto-fix? [Y/n]: {Colors.RESET}").strip().lower() != 'n'
                        run_single_file_analysis(project, file, current_provider, auto_fix)
                        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
            elif choice == '3':
                # Workflow Mode (Multi-Agent with fixed flow)
                project = select_project()
                if project:
                    run_multi_agent_analysis(project, current_provider)
                    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
            elif choice == '4':
                # True Agent (Autonomous mode)
                project = select_project()
                if project:
                    run_true_agent_analysis(project, current_provider)
                    input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
            elif choice == '5':
                # List Projects
                projects = get_projects()
                print_section("Available Projects")
                if projects:
                    for project in projects:
                        path = BENCH_DIR / project
                        py_files = len([f for f in path.rglob("*.py") if "__pycache__" not in str(f)])
                        print(f"  {Colors.BRIGHT_GREEN}{project}{Colors.RESET} {Colors.DIM}({py_files} files){Colors.RESET}")
                else:
                    print_status("warn", "No projects found")
                input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
            elif choice == '6':
                # Settings
                current_provider = settings_menu(current_provider)
                input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.DIM}Interrupted. Goodbye!{Colors.RESET}\n")
            break
        except Exception as e:
            print_status("error", f"Error: {str(e)}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CODE EVAL - Autonomous Multi-Agent Code Analysis System v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                         Interactive mode
  python cli.py scan myproject          Quick scan a project
  python cli.py workflow myproject      Multi-agent autonomous analysis
  python cli.py agent myproject         Same as workflow (unified system)
  python cli.py fix myproject file.py   Fix a single file

Autonomous Agent System:
  - 4 Core Agents: scanner, analyzer, fixer, executor
  - Supervisor LLM observes state and decides each step dynamically
  - Each agent has dedicated tools for their specific role
  - Dynamic routing based on error analysis

Both 'workflow' and 'agent' commands now use the same autonomous system.
        """
    )
    
    parser.add_argument('command', nargs='?', default='interactive',
                       choices=['interactive', 'scan', 'workflow', 'agent', 'fix', 'list'],
                       help='Command to run')
    parser.add_argument('project', nargs='?', help='Project name')
    parser.add_argument('file', nargs='?', help='File path (for fix command)')
    parser.add_argument('-p', '--provider', default='openrouter',
                       help='LLM provider (default: openrouter)')
    parser.add_argument('--no-fix', action='store_true',
                       help='Disable auto-fix for single file mode')
    parser.add_argument('--max-attempts', type=int, default=5,
                       help='Max fix attempts for workflow (default: 5)')
    parser.add_argument('--max-iterations', type=int, default=20,
                       help='Max iterations for True Agent (default: 20)')
    parser.add_argument('-r', '--request', type=str, default="Analyze the project and fix any code issues",
                       help='User request for True Agent')
    
    args = parser.parse_args()
    
    # Ensure code_bench exists
    BENCH_DIR.mkdir(exist_ok=True)
    
    if args.command == 'interactive' or args.command is None:
        interactive_mode()
    
    elif args.command == 'list':
        print_header("CODE EVAL", "Available Projects")
        projects = get_projects()
        if projects:
            for project in projects:
                path = BENCH_DIR / project
                py_files = len([f for f in path.rglob("*.py") if "__pycache__" not in str(f)])
                print(f"  {Colors.BRIGHT_GREEN}{project}{Colors.RESET} {Colors.DIM}({py_files} files){Colors.RESET}")
        else:
            print_status("warn", "No projects found in code_bench/")
    
    elif args.command == 'scan':
        if not args.project:
            print_status("error", "Project name required. Usage: cli.py scan <project>")
            return 1
        if args.project not in get_projects():
            print_status("error", f"Project not found: {args.project}")
            return 1
        run_quick_scan(args.project)
    
    elif args.command == 'workflow':
        # Workflow mode (fixed flow)
        if not args.project:
            print_status("error", "Project name required. Usage: cli.py workflow <project>")
            return 1
        if args.project not in get_projects():
            print_status("error", f"Project not found: {args.project}")
            return 1
        run_multi_agent_analysis(args.project, args.provider, args.max_attempts)
    
    elif args.command == 'agent':
        # True Agent mode (autonomous)
        if not args.project:
            print_status("error", "Project name required. Usage: cli.py agent <project>")
            return 1
        if args.project not in get_projects():
            print_status("error", f"Project not found: {args.project}")
            return 1
        run_true_agent_analysis(args.project, args.provider, args.max_iterations)
    
    elif args.command == 'fix':
        if not args.project or not args.file:
            print_status("error", "Project and file required. Usage: cli.py fix <project> <file>")
            return 1
        if args.project not in get_projects():
            print_status("error", f"Project not found: {args.project}")
            return 1
        run_single_file_analysis(args.project, args.file, args.provider, not args.no_fix)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

