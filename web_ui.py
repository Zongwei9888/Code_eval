#!/usr/bin/env python
"""
CODE EVAL v3.0 - Advanced Multi-Agent Code Analysis System
Tested and Working!
"""
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uuid
import traceback

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

# Single file workflow
from workflow.code_workflow_improved import create_workflow

# Repo workflow (new multi-agent system with feedback loops)
from agent.multi_agent_system import create_multi_agent_workflow, ALL_TOOLS as ALL_REPO_TOOLS

# Local utilities
from agent.repo_agents import quick_scan

from config import DEFAULT_PROVIDER, MODEL_MAPPINGS


# ============================================================================
# GLOBAL STATE
# ============================================================================
single_file_workflow = None
repo_workflow = None
BENCH_DIR = Path(__file__).parent / "code_bench"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_projects() -> List[str]:
    """Get available projects"""
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


def format_time():
    return datetime.now().strftime("%H:%M:%S")


def log_msg(logs: List[str], msg: str, level: str = "INFO"):
    """Add a log message"""
    logs.append(f"[{format_time()}] [{level}] {msg}")


# ============================================================================
# SINGLE FILE MODE HANDLERS
# ============================================================================

def load_file_content(project_name: str, file_path: str) -> str:
    """Load a file for editing"""
    if not project_name or not file_path:
        return "# Select a project and file first"
    
    full_path = BENCH_DIR / project_name / file_path
    if not full_path.exists():
        return f"# File not found: {file_path}"
    
    try:
        return full_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"# Error reading file: {str(e)}"


def run_single_file_analysis(project_name: str, file_path: str, provider: str, auto_fix: bool):
    """Run analysis on a single file"""
    global single_file_workflow
    
    if not project_name or not file_path:
        yield create_status("error", "Error", "Select a project and file"), create_logs([]), ""
        return
    
    full_path = BENCH_DIR / project_name / file_path
    if not full_path.exists():
        yield create_status("error", "Error", f"File not found: {file_path}"), create_logs([]), ""
        return
    
    logs = []
    log_msg(logs, f"Starting analysis on {file_path}")
    log_msg(logs, f"Provider: {provider}, Auto-fix: {auto_fix}")
    
    yield create_status("running", "Initializing", "Creating workflow..."), create_logs(logs), ""
    
    try:
        # Create workflow
        log_msg(logs, "Creating code improvement workflow...")
        single_file_workflow = create_workflow(
            llm_provider=provider, 
            max_attempts=5 if auto_fix else 1
        )
        
        # Read file
        original_code = full_path.read_text(encoding='utf-8')
        log_msg(logs, f"Loaded file: {len(original_code)} characters")
        
        yield create_status("running", "Analyzing", "Running agents..."), create_logs(logs), original_code
        
        # Run workflow with streaming to capture LLM responses
        thread_id = str(uuid.uuid4())
        log_msg(logs, f"Thread: {thread_id[:8]}...")
        
        # Stream workflow to get step-by-step updates
        step = 0
        final_result = None
        current_node = None
        for update in single_file_workflow.stream_run(str(full_path), original_code, thread_id=thread_id):
            step += 1
            if update:
                node_name = list(update.keys())[0]
                node_data = update[node_name]
                
                # Add step header when node changes
                if node_name != current_node:
                    agent_name = node_name.replace("_node", "").upper()
                    logs.append(f"Step {step}: [{agent_name}]")
                    current_node = node_name
                
                # Extract and display LLM response
                if isinstance(node_data, dict):
                    # Show code analysis (LLM response) in chat format
                    if node_data.get("code_analysis"):
                        analysis = node_data["code_analysis"][:400]
                        logs.append(f"[Analyzer] Turn {step}: {analysis}")
                    
                    # Show execution result
                    if "execution_success" in node_data:
                        success = node_data["execution_success"]
                        if success:
                            logs.append(f"[OK] Code executed successfully")
                        else:
                            error = node_data.get("last_error", "Unknown error")[:200]
                            logs.append(f"[Executor] Turn {step}: Execution failed - {error}")
                    
                    # Show modifications
                    if node_data.get("modification_history"):
                        mods = node_data["modification_history"]
                        if mods:
                            logs.append(f"[Fixer] Turn {step}: Applied code modification")
                    
                    final_result = node_data
                
                yield create_status("running", f"Step {step}", node_name), create_logs(logs), original_code
        
        # Process final results
        if final_result:
            success = final_result.get("execution_success", False)
            attempts = final_result.get("execution_attempts", 0)
            
            if success:
                log_msg(logs, f"SUCCESS after {attempts} attempt(s)", "OK")
            else:
                log_msg(logs, f"FAILED after {attempts} attempt(s)", "WARN")
            
            # Log issues
            issues = final_result.get("identified_issues", [])
            if issues:
                log_msg(logs, f"Found {len(issues)} issue(s):")
                for i, issue in enumerate(issues[:5], 1):
                    short_issue = issue[:100] + "..." if len(issue) > 100 else issue
                    logs.append(f"  {i}. {short_issue}")
        else:
            success = False
            attempts = 0
        
        # Get final code
        try:
            final_code = full_path.read_text(encoding='utf-8')
        except:
            final_code = final_result.get("current_code", original_code) if final_result else original_code
        
        if final_code != original_code:
            log_msg(logs, "Code was modified by agents", "OK")
        
        status = "success" if success else "warning"
        yield create_status(status, "Complete", f"{attempts} attempt(s)"), create_logs(logs), final_code
        
    except Exception as e:
        log_msg(logs, f"Error: {str(e)}", "ERROR")
        logs.append(traceback.format_exc())
        yield create_status("error", "Failed", str(e)[:50]), create_logs(logs), ""


def save_file(project_name: str, file_path: str, content: str) -> str:
    """Save modified file"""
    if not project_name or not file_path:
        return create_mini("error", "No file selected")
    
    full_path = BENCH_DIR / project_name / file_path
    
    try:
        full_path.write_text(content, encoding='utf-8')
        return create_mini("success", f"Saved: {file_path}")
    except Exception as e:
        return create_mini("error", f"Save failed: {str(e)}")


# ============================================================================
# REPO MODE HANDLERS
# ============================================================================

def run_quick_scan(project_name: str):
    """Run quick local scan"""
    if not project_name:
        return create_status("error", "Error", "Select a project"), "", ""
    
    project_path = BENCH_DIR / project_name
    if not project_path.exists():
        return create_status("error", "Error", "Project not found"), "", ""
    
    try:
        result = quick_scan(str(project_path))
        
        # Status
        has_errors = result["files_with_syntax_errors"] > 0
        status = create_status(
            "warning" if has_errors else "success",
            result["project_name"],
            f"{result['total_files']} files | {result['files_with_syntax_errors']} errors"
        )
        
        # Details card
        details = f'''
        <div style="background: #0d1117; padding: 20px; border-radius: 12px; border: 1px solid #30363d;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 15px;">
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #7ee787; font-size: 24px; font-weight: bold;">{result['total_files']}</div>
                    <div style="color: #8b949e; font-size: 11px;">PYTHON FILES</div>
                </div>
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: {'#f85149' if has_errors else '#7ee787'}; font-size: 24px; font-weight: bold;">{result['files_with_syntax_errors']}</div>
                    <div style="color: #8b949e; font-size: 11px;">SYNTAX ERRORS</div>
                </div>
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #58a6ff; font-size: 24px; font-weight: bold;">{'YES' if result['has_tests'] else 'NO'}</div>
                    <div style="color: #8b949e; font-size: 11px;">HAS TESTS</div>
                </div>
            </div>
        '''
        
        if result["syntax_errors"]:
            details += '<div style="color: #f85149; font-size: 12px; margin-bottom: 8px; font-weight: bold;">SYNTAX ERRORS:</div>'
            for err in result["syntax_errors"][:5]:
                errs = "; ".join(err["errors"][:2])
                details += f'''
                <div style="background: #21262d; padding: 8px; margin: 4px 0; border-radius: 6px; border-left: 3px solid #f85149;">
                    <div style="color: #f0f6fc; font-size: 11px; font-family: monospace;">{err["file"]}</div>
                    <div style="color: #f85149; font-size: 10px;">{errs}</div>
                </div>
                '''
        else:
            details += '<div style="color: #7ee787; text-align: center; padding: 20px;">All files have valid syntax!</div>'
        
        details += '</div>'
        
        # File tree
        tree = create_tree(project_path)
        
        return status, details, tree
        
    except Exception as e:
        return create_status("error", "Scan Failed", str(e)[:50]), f"<pre>{traceback.format_exc()}</pre>", ""


def run_repo_analysis(project_name: str, provider: str):
    """Run multi-agent repo analysis with feedback loops"""
    global repo_workflow
    
    if not project_name:
        yield create_status("error", "Error", "Select a project"), create_logs([]), ""
        return
    
    project_path = BENCH_DIR / project_name
    if not project_path.exists():
        yield create_status("error", "Error", "Project not found"), create_logs([]), ""
        return
    
    logs = []
    log_msg(logs, f"Starting multi-agent analysis on {project_name}")
    log_msg(logs, f"Provider: {provider}")
    
    yield create_status("running", "Initializing", "Creating workflow..."), create_logs(logs), ""
    
    try:
        log_msg(logs, "Creating Multi-Agent Workflow with feedback loops...")
        log_msg(logs, "Flow: Scanner -> Analyzer -> [Fixer <-> Executor] -> Reporter")
        repo_workflow = create_multi_agent_workflow(provider, max_fix_attempts=5)
        
        yield create_status("running", "Running", "Multi-agent pipeline..."), create_logs(logs), ""
        
        # Stream execution
        log_msg(logs, "Starting stream execution...")
        step_count = 0
        final_state = None
        all_step_logs = []
        current_agent = None
        
        for update in repo_workflow.stream_run(str(project_path)):
            step_count += 1
            if update:
                node = list(update.keys())[0]
                node_data = update.get(node, {})
                
                # Add step header when agent changes
                if node != current_agent:
                    logs.append(f"Step {step_count}: [{node.upper()}]")
                    current_agent = node
                
                # Extract step logs from new system
                if isinstance(node_data, dict):
                    step_logs = node_data.get("step_logs", [])
                    
                    # Show new logs only
                    new_logs = step_logs[len(all_step_logs):]
                    all_step_logs = step_logs
                    
                    for sl in new_logs:
                        agent = sl.get("agent", "")
                        turn = sl.get("turn", 0)
                        log_type = sl.get("type", "")
                        
                        if log_type == "llm_response":
                            content = sl.get("content", "")[:300]
                            has_tools = sl.get("has_tool_calls", False)
                            # Format for chat bubble parsing
                            tools_suffix = " (calling tools)" if has_tools else ""
                            logs.append(f"[{agent}] Turn {turn}: {content}{tools_suffix}")
                        elif log_type == "tool_call":
                            tool = sl.get("tool", "unknown")
                            result = sl.get("result", "")[:150]
                            logs.append(f"[Tool] {tool}: {result}")
                    
                    # Show key state changes
                    if node_data.get("files_with_errors"):
                        errs = node_data["files_with_errors"]
                        logs.append(f"[!] Found {len(errs)} file(s) with errors")
                    
                    if "fix_attempts" in node_data and node_data["fix_attempts"] > 0:
                        logs.append(f"[INFO] Fix attempt #{node_data['fix_attempts']}")
                    
                    if "last_execution_success" in node_data:
                        success = node_data["last_execution_success"]
                        if success:
                            logs.append(f"[OK] Execution SUCCESS")
                        else:
                            logs.append(f"[ERROR] Execution FAILED")
                
                final_state = node_data
            
            yield create_status("running", f"Step {step_count}", node.upper() if update else "processing"), create_logs(logs), ""
            
            if step_count > 30:
                log_msg(logs, "Reached iteration limit", "WARN")
                break
        
        log_msg(logs, f"Workflow complete in {step_count} steps", "OK")
        
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
        
        # Check final status
        exec_success = final_state.get("last_execution_success", False) if final_state else False
        fix_attempts = final_state.get("fix_attempts", 0) if final_state else 0
        
        status_color = "#7ee787" if exec_success else "#d29922"
        status_text = "SUCCESS" if exec_success else f"INCOMPLETE ({fix_attempts} fixes)"
        
        # Create report
        report = f'''
        <div style="background: #0d1117; padding: 20px; border-radius: 12px; border: 1px solid #30363d;">
            <div style="color: {status_color}; font-size: 18px; text-align: center; margin-bottom: 15px; font-weight: bold;">
                {status_text}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #7ee787; font-size: 20px; font-weight: bold;">{step_count}</div>
                    <div style="color: #8b949e; font-size: 10px;">STEPS</div>
                </div>
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #58a6ff; font-size: 20px; font-weight: bold;">{len(agents_seen)}</div>
                    <div style="color: #8b949e; font-size: 10px;">AGENTS</div>
                </div>
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #a371f7; font-size: 20px; font-weight: bold;">{llm_calls}</div>
                    <div style="color: #8b949e; font-size: 10px;">LLM CALLS</div>
                </div>
                <div style="background: #161b22; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #f78166; font-size: 20px; font-weight: bold;">{tool_calls}</div>
                    <div style="color: #8b949e; font-size: 10px;">TOOLS</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #161b22; border-radius: 8px;">
                <div style="color: #8b949e; font-size: 11px;">
                    <div>Project: {project_name}</div>
                    <div>Fix Attempts: {fix_attempts}</div>
                    <div>Agents: {", ".join(sorted(agents_seen))}</div>
                </div>
            </div>
        </div>
        '''
        
        status = "success" if exec_success else "warning"
        yield create_status(status, "Complete", f"{llm_calls} LLM calls, {tool_calls} tools"), create_logs(logs), report
        
    except Exception as e:
        log_msg(logs, f"Error: {str(e)}", "ERROR")
        logs.append(traceback.format_exc())
        yield create_status("error", "Failed", str(e)[:50]), create_logs(logs), ""


# ============================================================================
# HTML COMPONENTS
# ============================================================================

def create_status(status: str, title: str, subtitle: str) -> str:
    colors = {
        "success": ("#7ee787", "#238636"),
        "warning": ("#d29922", "#9e6a03"),
        "error": ("#f85149", "#da3633"),
        "running": ("#58a6ff", "#1f6feb"),
        "idle": ("#8b949e", "#484f58")
    }
    text_color, bg_color = colors.get(status, colors["idle"])
    icons = {"success": "[OK]", "warning": "[!]", "error": "[X]", "running": "[~]", "idle": "[-]"}
    icon = icons.get(status, "[-]")
    
    return f'''
    <div style="background: linear-gradient(135deg, {bg_color}22 0%, {bg_color}11 100%); 
                border: 1px solid {bg_color}; border-radius: 12px; padding: 20px; text-align: center;">
        <div style="color: {text_color}; font-size: 24px; font-weight: bold;">{icon} {title}</div>
        <div style="color: #8b949e; font-size: 12px; margin-top: 5px;">{subtitle}</div>
    </div>
    '''


def create_mini(status: str, message: str) -> str:
    colors = {"success": "#7ee787", "error": "#f85149", "warning": "#d29922"}
    return f'<div style="color: {colors.get(status, "#8b949e")}; font-size: 12px; padding: 5px;">{message}</div>'


def create_chat_bubble(role: str, content: str, turn: int = 0, has_tools: bool = False, tool_name: str = "") -> str:
    """Create a chat-style bubble for messages"""
    
    # Role styles
    role_styles = {
        "llm": ("#a371f7", "#a371f722", "LLM", "left"),
        "scanner": ("#58a6ff", "#58a6ff22", "SCANNER", "left"),
        "analyzer": ("#d29922", "#d2992222", "ANALYZER", "left"),
        "fixer": ("#7ee787", "#7ee78722", "FIXER", "left"),
        "executor": ("#79c0ff", "#79c0ff22", "EXECUTOR", "left"),
        "reporter": ("#a371f7", "#a371f722", "REPORTER", "left"),
        "tool": ("#f78166", "#f7816622", "TOOL", "right"),
        "error": ("#f85149", "#f8514922", "ERROR", "left"),
        "system": ("#8b949e", "#8b949e22", "SYSTEM", "center"),
    }
    
    color, bg, label, align = role_styles.get(role.lower(), ("#c9d1d9", "transparent", role.upper(), "left"))
    
    # Escape HTML in content
    content_escaped = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    # Turn indicator
    turn_html = f'<span style="color: #484f58; font-size: 9px;">T{turn}</span>' if turn > 0 else ""
    
    # Tool indicator
    tool_html = f'<span style="color: #d29922; font-size: 9px; margin-left: 5px;">&#8594; tools</span>' if has_tools else ""
    
    # Tool name for tool messages
    tool_name_html = f'<span style="color: #58a6ff; font-size: 10px;">{tool_name}</span> ' if tool_name else ""
    
    margin = "margin-left: auto;" if align == "right" else ("margin: 0 auto;" if align == "center" else "")
    max_width = "85%" if align != "center" else "95%"
    
    return f'''
    <div style="display: flex; margin: 8px 0;">
        <div style="background: {bg}; border: 1px solid {color}44; border-radius: 12px; 
                    padding: 10px 14px; max-width: {max_width}; {margin}">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; border-bottom: 1px solid {color}33; padding-bottom: 6px;">
                <span style="background: {color}; color: #0d1117; padding: 2px 8px; border-radius: 4px; 
                            font-size: 10px; font-weight: bold;">{label}</span>
                {turn_html}
                {tool_html}
                {tool_name_html}
            </div>
            <div style="color: #c9d1d9; font-size: 12px; line-height: 1.5; word-wrap: break-word;">
                {content_escaped[:600]}{"..." if len(content) > 600 else ""}
            </div>
        </div>
    </div>
    '''


def create_step_header(step: int, agent: str) -> str:
    """Create a step header divider"""
    agent_colors = {
        "scanner": "#58a6ff",
        "analyzer": "#d29922",
        "fixer": "#7ee787",
        "executor": "#79c0ff",
        "reporter": "#a371f7",
    }
    color = agent_colors.get(agent.lower(), "#58a6ff")
    
    return f'''
    <div style="display: flex; align-items: center; margin: 15px 0 10px 0;">
        <div style="flex: 1; height: 1px; background: linear-gradient(90deg, transparent, {color}66);"></div>
        <div style="background: {color}22; border: 1px solid {color}44; padding: 4px 12px; border-radius: 6px; margin: 0 10px;">
            <span style="color: {color}; font-size: 11px; font-weight: bold;">STEP {step}</span>
            <span style="color: #8b949e; font-size: 11px;"> | </span>
            <span style="color: {color}; font-size: 11px;">{agent.upper()}</span>
        </div>
        <div style="flex: 1; height: 1px; background: linear-gradient(90deg, {color}66, transparent);"></div>
    </div>
    '''


def create_logs(logs: List[str]) -> str:
    """Create styled log container with chat-like messages"""
    html = ""
    current_step = 0
    
    for log in logs[-50:]:
        # Check for step markers
        if "Step " in log and ":" in log:
            try:
                # Extract step number and agent
                parts = log.split("Step ")[1].split(":")
                step_num = int(parts[0].strip())
                if step_num != current_step:
                    current_step = step_num
                    agent = parts[1].strip().replace("[", "").replace("]", "").split()[0] if len(parts) > 1 else "AGENT"
                    html += create_step_header(step_num, agent)
                continue
            except:
                pass
        
        # Parse different log types
        if "[ERROR]" in log or "Error:" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += create_chat_bubble("error", content)
        elif "[OK]" in log or "SUCCESS" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div style="color: #7ee787; font-size: 11px; padding: 4px 10px; margin: 4px 0; background: #7ee78711; border-radius: 6px; text-align: center;">{content}</div>'
        elif "[WARN]" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div style="color: #d29922; font-size: 11px; padding: 4px 10px; margin: 4px 0; background: #d2992211; border-radius: 6px;">{content}</div>'
        elif log.strip().startswith("[") and "] Turn" in log:
            # Agent message with turn
            try:
                agent = log.split("[")[1].split("]")[0]
                turn = int(log.split("Turn ")[1].split(":")[0])
                content = log.split(":", 2)[-1].strip() if ":" in log else log
                has_tools = "-> Will call tools" in log or "tools" in content.lower()
                html += create_chat_bubble(agent.lower(), content, turn, has_tools)
            except:
                html += f'<div style="color: #c9d1d9; font-size: 11px; padding: 2px 8px;">{log}</div>'
        elif "[Tool]" in log:
            # Tool call
            try:
                parts = log.split("[Tool]")[1].strip()
                tool_name = parts.split(":")[0].strip()
                result = parts.split(":", 1)[1].strip() if ":" in parts else ""
                html += create_chat_bubble("tool", result, tool_name=tool_name)
            except:
                html += f'<div style="color: #f78166; font-size: 11px; padding: 2px 8px;">{log}</div>'
        elif "[INFO]" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div style="color: #58a6ff; font-size: 10px; padding: 3px 10px; margin: 2px 0;">{content}</div>'
        elif log.strip().startswith("[!]"):
            content = log.replace("[!]", "").strip()
            html += f'<div style="color: #f85149; font-size: 11px; padding: 4px 10px; margin: 4px 0; background: #f8514911; border-radius: 6px; border-left: 3px solid #f85149;">{content}</div>'
        elif log.strip():
            # Default log style
            html += f'<div style="color: #8b949e; font-size: 10px; padding: 2px 8px; font-family: monospace;">{log}</div>'
    
    return f'''
    <div style="background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); 
                padding: 15px; border-radius: 12px; border: 1px solid #30363d; 
                height: 450px; overflow-y: auto; box-shadow: inset 0 2px 10px rgba(0,0,0,0.3);">
        <div style="display: flex; flex-direction: column; gap: 2px;">
            {html if html else '<div style="color: #484f58; text-align: center; padding: 20px;">Waiting for agent activity...</div>'}
        </div>
    </div>
    '''


def create_tree(project_path: Path) -> str:
    html = '''<div style="background: #0d1117; padding: 15px; border-radius: 10px; border: 1px solid #30363d; 
                         font-family: monospace; font-size: 11px; max-height: 300px; overflow-y: auto;">'''
    
    def add_items(path: Path, indent: int = 0):
        nonlocal html
        prefix = "&nbsp;" * (indent * 3)
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items[:50]:
                if item.name.startswith('.') or item.name == '__pycache__':
                    continue
                if item.is_dir():
                    html += f'<div style="color: #58a6ff;">{prefix}[+] {item.name}/</div>'
                    if indent < 2:
                        add_items(item, indent + 1)
                else:
                    color = "#7ee787" if item.suffix == ".py" else "#8b949e"
                    html += f'<div style="color: {color};">{prefix}    {item.name}</div>'
        except:
            pass
    
    add_items(project_path)
    html += '</div>'
    return html


# ============================================================================
# UI BUILDER
# ============================================================================

def create_ui():
    # Get initial data
    initial_projects = get_projects()
    
    with gr.Blocks(title="CODE EVAL v3") as app:
        
        # Header
        gr.HTML('''
        <div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); 
                    padding: 30px 20px; text-align: center; border-bottom: 1px solid #30363d;">
            <div style="font-size: 36px; font-weight: 800; letter-spacing: 4px;
                        background: linear-gradient(90deg, #58a6ff, #7ee787, #58a6ff);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                CODE EVAL
            </div>
            <div style="color: #8b949e; font-size: 12px; margin-top: 5px; letter-spacing: 2px;">
                MULTI-AGENT CODE ANALYSIS | TESTED AND WORKING
            </div>
        </div>
        ''')
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: SINGLE FILE MODE
            # ============================================================
            with gr.TabItem("Single File Analysis"):
                gr.HTML('''
                <div style="background: #161b22; color: #58a6ff; padding: 10px 15px; font-size: 12px; border-radius: 8px; margin: 10px 0;">
                    Analyze and auto-fix individual Python files using CodeAnalyzer -> CodeExecutor -> CodeModifier agents
                </div>
                ''')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sf_project = gr.Dropdown(
                            label="Select Project",
                            choices=initial_projects,
                            value=initial_projects[0] if initial_projects else None,
                            interactive=True
                        )
                        sf_file = gr.Dropdown(
                            label="Select File",
                            choices=get_files_in_project(initial_projects[0]) if initial_projects else [],
                            interactive=True
                        )
                        sf_provider = gr.Dropdown(
                            label="LLM Provider",
                            choices=list(MODEL_MAPPINGS.keys()),
                            value=DEFAULT_PROVIDER
                        )
                        sf_auto_fix = gr.Checkbox(label="Auto-Fix Mode (iterate until success)", value=True)
                        sf_run_btn = gr.Button("ANALYZE & FIX", variant="primary", size="lg")
                        sf_refresh = gr.Button("Refresh Projects", size="sm")
                    
                    with gr.Column(scale=2):
                        sf_status = gr.HTML(value=create_status("idle", "Ready", "Select a file to analyze"))
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sf_logs = gr.HTML(value=create_logs([]))
                    
                    with gr.Column(scale=1):
                        sf_code = gr.Code(
                            label="Code Editor",
                            language="python",
                            lines=20,
                            interactive=True
                        )
                        with gr.Row():
                            sf_save_btn = gr.Button("SAVE FILE", variant="primary", size="sm")
                            sf_save_status = gr.HTML()
            
            # ============================================================
            # TAB 2: QUICK SCAN
            # ============================================================
            with gr.TabItem("Quick Scan"):
                gr.HTML('''
                <div style="background: #161b22; color: #58a6ff; padding: 10px 15px; font-size: 12px; border-radius: 8px; margin: 10px 0;">
                    Fast local scan using Python AST - No AI required, instant results
                </div>
                ''')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        qs_project = gr.Dropdown(
                            label="Select Project",
                            choices=initial_projects,
                            value=initial_projects[0] if initial_projects else None,
                            interactive=True
                        )
                        qs_scan_btn = gr.Button("QUICK SCAN", variant="primary", size="lg")
                        qs_refresh = gr.Button("Refresh Projects", size="sm")
                    
                    with gr.Column(scale=2):
                        qs_status = gr.HTML(value=create_status("idle", "Ready", "Select project and scan"))
                
                with gr.Row():
                    qs_details = gr.HTML()
                    qs_tree = gr.HTML()
            
            # ============================================================
            # TAB 3: MULTI-AGENT ANALYSIS
            # ============================================================
            with gr.TabItem("Multi-Agent Analysis"):
                gr.HTML('''
                <div style="background: #161b22; color: #58a6ff; padding: 10px 15px; font-size: 12px; border-radius: 8px; margin: 10px 0;">
                    Full LangGraph workflow: Scanner -> Analyzer -> Tester -> Fixer -> Reporter
                </div>
                ''')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ma_project = gr.Dropdown(
                            label="Select Project",
                            choices=initial_projects,
                            value=initial_projects[0] if initial_projects else None,
                            interactive=True
                        )
                        ma_provider = gr.Dropdown(
                            label="LLM Provider",
                            choices=list(MODEL_MAPPINGS.keys()),
                            value=DEFAULT_PROVIDER
                        )
                        ma_run_btn = gr.Button("START ANALYSIS", variant="primary", size="lg")
                        ma_refresh = gr.Button("Refresh Projects", size="sm")
                    
                    with gr.Column(scale=2):
                        ma_status = gr.HTML(value=create_status("idle", "Ready", "Configure and start"))
                
                with gr.Row():
                    ma_logs = gr.HTML(value=create_logs([]))
                    ma_report = gr.HTML()
            
            # ============================================================
            # TAB 4: HELP
            # ============================================================
            with gr.TabItem("Help"):
                gr.HTML('''
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                    <div style="color: #58a6ff; font-size: 20px; margin-bottom: 15px;">How to Use CODE EVAL</div>
                    
                    <div style="background: #161b22; padding: 15px; border-radius: 10px; margin-bottom: 12px; border-left: 3px solid #7ee787;">
                        <div style="color: #7ee787; font-size: 14px; margin-bottom: 5px;">1. Single File Analysis</div>
                        <div style="color: #8b949e; font-size: 12px;">
                            Fixes individual files. Agents: CodeAnalyzer -> CodeExecutor -> CodeModifier.<br>
                            Enable "Auto-Fix Mode" for iterative fixing until code runs successfully.
                        </div>
                    </div>
                    
                    <div style="background: #161b22; padding: 15px; border-radius: 10px; margin-bottom: 12px; border-left: 3px solid #58a6ff;">
                        <div style="color: #58a6ff; font-size: 14px; margin-bottom: 5px;">2. Quick Scan</div>
                        <div style="color: #8b949e; font-size: 12px;">
                            Instant local analysis using Python AST. No AI calls.<br>
                            Detects syntax errors and shows project structure.
                        </div>
                    </div>
                    
                    <div style="background: #161b22; padding: 15px; border-radius: 10px; margin-bottom: 12px; border-left: 3px solid #d29922;">
                        <div style="color: #d29922; font-size: 14px; margin-bottom: 5px;">3. Multi-Agent Analysis</div>
                        <div style="color: #8b949e; font-size: 12px;">
                            Full LangGraph workflow with 5 specialized agents:<br>
                            <span style="color: #58a6ff;">Scanner</span> -> 
                            <span style="color: #58a6ff;">Analyzer</span> -> 
                            <span style="color: #58a6ff;">Tester</span> -> 
                            <span style="color: #58a6ff;">Fixer</span> -> 
                            <span style="color: #58a6ff;">Reporter</span>
                        </div>
                    </div>
                    
                    <div style="background: #21262d; padding: 12px; border-radius: 8px;">
                        <div style="color: #f85149; font-size: 12px; margin-bottom: 5px;">Setup</div>
                        <div style="color: #8b949e; font-size: 11px;">
                            1. Place your projects in: <code style="color: #58a6ff;">code_bench/</code><br>
                            2. Set your API key in: <code style="color: #58a6ff;">config/llm_config.py</code>
                        </div>
                    </div>
                </div>
                ''')
        
        # Footer
        gr.HTML('''
        <div style="text-align: center; padding: 12px; border-top: 1px solid #30363d; margin-top: 20px;">
            <div style="color: #484f58; font-size: 10px;">CODE EVAL v3.0 | Multi-Agent Code Analysis System</div>
        </div>
        ''')
        
        # ============================================================
        # EVENT HANDLERS
        # ============================================================
        
        def update_file_list(project):
            files = get_files_in_project(project)
            return gr.update(choices=files, value=files[0] if files else None)
        
        def refresh_projects():
            projects = get_projects()
            return gr.update(choices=projects, value=projects[0] if projects else None)
        
        # Single File Mode
        sf_project.change(fn=update_file_list, inputs=[sf_project], outputs=[sf_file])
        sf_file.change(fn=load_file_content, inputs=[sf_project, sf_file], outputs=[sf_code])
        sf_run_btn.click(
            fn=run_single_file_analysis,
            inputs=[sf_project, sf_file, sf_provider, sf_auto_fix],
            outputs=[sf_status, sf_logs, sf_code]
        )
        sf_save_btn.click(fn=save_file, inputs=[sf_project, sf_file, sf_code], outputs=[sf_save_status])
        sf_refresh.click(fn=refresh_projects, outputs=[sf_project])
        
        # Quick Scan Mode
        qs_scan_btn.click(fn=run_quick_scan, inputs=[qs_project], outputs=[qs_status, qs_details, qs_tree])
        qs_refresh.click(fn=refresh_projects, outputs=[qs_project])
        
        # Multi-Agent Mode
        ma_run_btn.click(
            fn=run_repo_analysis,
            inputs=[ma_project, ma_provider],
            outputs=[ma_status, ma_logs, ma_report]
        )
        ma_refresh.click(fn=refresh_projects, outputs=[ma_project])
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    BENCH_DIR.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  CODE EVAL v3.0 - Multi-Agent Code Analysis")
    print("=" * 60)
    print(f"  Projects folder: {BENCH_DIR}")
    print(f"  Available projects: {get_projects()}")
    print("=" * 60 + "\n")
    
    app = create_ui()
    app.launch(server_name="127.0.0.1", share=False, inbrowser=True)
