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
from workflow import create_workflow

# Repo workflow (multi-agent system with feedback loops)
from workflow import create_multi_agent_workflow

# Local utilities (non-LLM based)
from agent import quick_scan

# Tools (for reference)
from tools import ALL_REPO_TOOLS

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
        <div class="card-container" style="padding: 20px;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;">
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--success);">{result['total_files']}</div>
                    <div class="metric-label">PYTHON FILES</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: {'var(--error)' if has_errors else 'var(--success)'};">{result['files_with_syntax_errors']}</div>
                    <div class="metric-label">SYNTAX ERRORS</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--primary);">{ 'YES' if result['has_tests'] else 'NO' }</div>
                    <div class="metric-label">HAS TESTS</div>
                </div>
            </div>
        '''
        
        if result["syntax_errors"]:
            details += '<div style="color: var(--error); font-size: 12px; margin-bottom: 8px; font-weight: 600;">SYNTAX ERRORS:</div>'
            for err in result["syntax_errors"][:5]:
                errs = "; ".join(err["errors"][:2])
                details += f'''
                <div style="background: rgba(248, 81, 73, 0.1); padding: 8px; margin: 4px 0; border-radius: 6px; border-left: 3px solid var(--error);">
                    <div style="color: #e2e8f0; font-size: 11px; font-family: monospace;">{err["file"]}</div>
                    <div style="color: var(--error); font-size: 10px;">{errs}</div>
                </div>
                '''
        else:
            details += '<div style="color: var(--success); text-align: center; padding: 20px;">All files have valid syntax!</div>'
        
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
        
        status_color = "var(--success)" if exec_success else "var(--warning)"
        status_text = "SUCCESS" if exec_success else f"INCOMPLETE ({fix_attempts} fixes)"
        
        # Create report
        report = f'''
        <div class="card-container" style="padding: 20px;">
            <div style="color: {status_color}; font-size: 18px; text-align: center; margin-bottom: 15px; font-weight: 700;">
                {status_text}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--success);">{step_count}</div>
                    <div class="metric-label">STEPS</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--primary);">{len(agents_seen)}</div>
                    <div class="metric-label">AGENTS</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--llm);">{llm_calls}</div>
                    <div class="metric-label">LLM CALLS</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: var(--tool);">{tool_calls}</div>
                    <div class="metric-label">TOOLS</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 11px; color: #94a3b8;">
                <div>Project: <span style="color: #e2e8f0;">{project_name}</span></div>
                <div>Fix Attempts: <span style="color: #e2e8f0;">{fix_attempts}</span></div>
                <div>Agents: <span style="color: #e2e8f0;">{", ".join(sorted(agents_seen))}</span></div>
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
# HTML COMPONENTS (MODERN THEME)
# ============================================================================

def create_status(status: str, title: str, subtitle: str) -> str:
    status_map = {
        "success": ("var(--success)", "[OK]"),
        "warning": ("var(--warning)", "[!]"),
        "error": ("var(--error)", "[X]"),
        "running": ("var(--primary)", "[~]"),
        "idle": ("#64748b", "[-]")
    }
    color, icon = status_map.get(status, status_map["idle"])
    
    # Convert var to actual color or handle opacity via style override
    style = f"border: 1px solid {color}; background: rgba(255,255,255,0.03);"
    if "var" in color:
        # Let CSS handle it mostly, but inject color for border
        pass
    
    return f'''
    <div style="background: rgba(255,255,255,0.03); border: 1px solid {color}; 
                border-radius: 12px; padding: 16px; display: flex; align-items: center; gap: 15px;">
        <div style="color: {color}; font-size: 24px; font-weight: 700;">{icon}</div>
        <div>
            <div style="color: var(--text-primary); font-size: 16px; font-weight: 600;">{title}</div>
            <div style="color: var(--text-dim); font-size: 12px; margin-top: 2px;">{subtitle}</div>
        </div>
    </div>
    '''


def create_mini(status: str, message: str) -> str:
    colors = {"success": "var(--success)", "error": "var(--error)", "warning": "var(--warning)"}
    return f'<div style="color: {colors.get(status, "#94a3b8")}; font-size: 12px; padding: 5px; font-weight: 500;">{message}</div>'


def create_chat_bubble(role: str, content: str, turn: int = 0, has_tools: bool = False, tool_name: str = "") -> str:
    """Create a sleek chat bubble"""
    
    # Modern Palette
    styles = {
        "llm":      ("var(--llm)",      "left"),
        "scanner":  ("var(--scanner)",  "left"),
        "analyzer": ("var(--analyzer)", "left"),
        "fixer":    ("var(--fixer)",    "left"),
        "executor": ("var(--executor)", "left"),
        "reporter": ("var(--reporter)", "left"),
        "tool":     ("var(--tool)",     "right"),
        "error":    ("var(--error)",    "left"),
        "system":   ("#64748b",         "center"),
    }
    
    color, align = styles.get(role.lower(), ("#94a3b8", "left"))
    
    content_escaped = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    turn_badge = f'<span class="turn-badge">T{turn}</span>' if turn > 0 else ""
    tool_badge = f'<span class="tool-badge">TOOLS</span>' if has_tools else ""
    tool_name_span = f'<span style="color: #60a5fa; font-size: 11px; margin-left: 8px;">{tool_name}</span>' if tool_name else ""
    
    is_right = align == "right"
    container_class = "chat-bubble-right" if is_right else ("chat-bubble-center" if align == "center" else "chat-bubble-left")
    
    return f'''
    <div class="chat-row {align}">
        <div class="chat-bubble {container_class}" style="border-left: 3px solid {color};">
            <div class="chat-header">
                <span style="color: {color}; font-weight: 700; font-size: 11px; text-transform: uppercase;">{role}</span>
                {turn_badge}
                {tool_badge}
                {tool_name_span}
            </div>
            <div class="chat-content">
                {content_escaped[:800]}{"..." if len(content) > 800 else ""}
            </div>
        </div>
    </div>
    '''


def create_step_header(step: int, agent: str) -> str:
    agent_colors = {
        "scanner": "var(--scanner)",
        "analyzer": "var(--analyzer)",
        "fixer": "var(--fixer)",
        "executor": "var(--executor)",
        "reporter": "var(--reporter)",
    }
    color = agent_colors.get(agent.lower(), "var(--primary)")
    
    return f'''
    <div class="step-divider">
        <div class="line" style="background: linear-gradient(90deg, transparent, {color}66);"></div>
        <div class="badge" style="border-color: {color}44; background: {color}11;">
            <span style="color: {color}; font-weight: 700;">STEP {step}</span>
            <span style="color: #64748b;"> | </span>
            <span style="color: {color};">{agent.upper()}</span>
        </div>
        <div class="line" style="background: linear-gradient(90deg, {color}66, transparent);"></div>
    </div>
    '''


def create_logs(logs: List[str]) -> str:
    html = ""
    current_step = 0
    
    for log in logs[-60:]: # Show more logs
        if "Step " in log and ":" in log:
            try:
                parts = log.split("Step ")[1].split(":")
                step_num = int(parts[0].strip())
                if step_num != current_step:
                    current_step = step_num
                    agent = parts[1].strip().replace("[", "").replace("]", "").split()[0] if len(parts) > 1 else "AGENT"
                    html += create_step_header(step_num, agent)
                continue
            except: pass
        
        if "[ERROR]" in log or "Error:" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += create_chat_bubble("error", content)
        elif "[OK]" in log or "SUCCESS" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div class="log-item success"><span>✓</span> {content}</div>'
        elif "[WARN]" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div class="log-item warning"><span>!</span> {content}</div>'
        elif log.strip().startswith("[") and "] Turn" in log:
            try:
                agent = log.split("[")[1].split("]")[0]
                turn = int(log.split("Turn ")[1].split(":")[0])
                content = log.split(":", 2)[-1].strip() if ":" in log else log
                has_tools = "-> Will call tools" in log or "tools" in content.lower()
                html += create_chat_bubble(agent.lower(), content, turn, has_tools)
            except:
                html += f'<div class="log-item info">{log}</div>'
        elif "[Tool]" in log:
            try:
                parts = log.split("[Tool]")[1].strip()
                tool_name = parts.split(":")[0].strip()
                result = parts.split(":", 1)[1].strip() if ":" in parts else ""
                html += create_chat_bubble("tool", result, tool_name=tool_name)
            except:
                html += f'<div class="log-item tool">{log}</div>'
        elif "[INFO]" in log:
            content = log.split("]", 1)[-1].strip() if "]" in log else log
            html += f'<div class="log-item info">{content}</div>'
        elif log.strip().startswith("[!]"):
            content = log.replace("[!]", "").strip()
            html += f'<div class="log-item error"><span>!</span> {content}</div>'
        elif log.strip():
            html += f'<div class="log-item dim">{log}</div>'
    
    return f'''
    <div class="log-container">
        <div class="log-scroll-area">
            {html if html else '<div class="empty-state">Ready for analysis...</div>'}
        </div>
    </div>
    '''


def create_tree(project_path: Path) -> str:
    html = '<div class="file-tree">'
    
    def add_items(path: Path, indent: int = 0):
        nonlocal html
        prefix = '<span class="indent"></span>' * indent
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items[:50]:
                if item.name.startswith('.') or item.name == '__pycache__':
                    continue
                if item.is_dir():
                    html += f'<div class="tree-item dir">{prefix}📁 {item.name}</div>'
                    if indent < 2:
                        add_items(item, indent + 1)
                else:
                    icon = "🐍" if item.suffix == ".py" else "📄"
                    html += f'<div class="tree-item file">{prefix}{icon} {item.name}</div>'
        except: pass
    
    add_items(project_path)
    html += '</div>'
    return html


# ============================================================================
# UI BUILDER
# ============================================================================

def create_ui():
    # CSS Variables & Global Styles
    css = """
    :root {
        --bg-dark: #09090b;
        --bg-card: #18181b;
        --bg-input: #27272a;
        --border-color: #3f3f46;
        
        /* High Contrast Text */
        --text-primary: #f8fafc;   /* Slate-50 */
        --text-secondary: #cbd5e1; /* Slate-300 */
        --text-dim: #94a3b8;       /* Slate-400 */
        
        --primary: #3b82f6;    /* Blue */
        --success: #10b981;    /* Emerald */
        --warning: #f59e0b;    /* Amber */
        --error: #ef4444;      /* Red */
        
        /* Agent Colors */
        --llm: #8b5cf6;        /* Violet */
        --scanner: #06b6d4;    /* Cyan */
        --analyzer: #f59e0b;   /* Amber */
        --fixer: #10b981;      /* Emerald */
        --executor: #3b82f6;   /* Blue */
        --reporter: #d946ef;   /* Fuchsia */
        --tool: #f97316;       /* Orange */
    }

    /* FORCE DARK MODE EVERYTHING */
    body, .gradio-container, .gradio-container * {
        background-color: var(--bg-dark) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Override Gradio Components specifically to fix white backgrounds */
    
    /* Inputs, Textareas, Dropdowns */
    textarea, input, .gr-input, .gr-text-input, .gr-box {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Dropdown specific fixes */
    .wrap-inner, .single-select, .selector-head, .selector-item, ul.options {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Buttons */
    button {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    button.primary {
        background: linear-gradient(135deg, var(--primary) 0%, #2563eb 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Panels and Boxes */
    .block, .panel, .tabs {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Tab Headers */
    .tab-nav {
        background-color: var(--bg-dark) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    .tab-nav button {
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
    }
    .tab-nav button.selected {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Code Editor Fixes */
    .cm-editor, .cm-scroller, .cm-gutters {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    .cm-line { color: #e6edf3 !important; }

    /* --- CUSTOM COMPONENTS --- */

    /* Card Containers */
    .card-container {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Metric Boxes */
    .metric-box {
        background: rgba(255,255,255,0.03) !important;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-val { font-size: 20px; font-weight: 700; margin-bottom: 4px; color: var(--text-primary) !important; }
    .metric-label { font-size: 10px; color: var(--text-dim) !important; letter-spacing: 1px; }

    /* Chat/Log Styles */
    .log-container {
        background: #0c0e11 !important;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        height: 500px;
        position: relative;
        overflow: hidden;
    }
    .log-scroll-area {
        height: 100%;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .empty-state {
        color: var(--text-dim) !important;
        text-align: center;
        margin-top: 100px;
        font-style: italic;
    }
    
    /* Step Divider */
    .step-divider {
        display: flex;
        align-items: center;
        margin: 24px 0 16px 0;
        background: transparent !important;
    }
    .step-divider .line { flex: 1; height: 1px; }
    .step-divider .badge {
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid;
        margin: 0 12px;
        font-size: 11px;
        letter-spacing: 0.5px;
        background: transparent !important; /* Badge bg handled inline */
    }
    
    /* Chat Bubbles */
    .chat-row { display: flex; width: 100%; margin: 6px 0; background: transparent !important; }
    .chat-row.right { justify-content: flex-end; }
    .chat-row.center { justify-content: center; }
    
    .chat-bubble {
        max-width: 85%;
        background: var(--bg-input) !important;
        padding: 12px 16px;
        border-radius: 12px;
        position: relative;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .chat-bubble-right { background: rgba(249, 115, 22, 0.1) !important; max-width: 75%; }
    
    .chat-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 6px;
        background: transparent !important;
    }
    .turn-badge {
        background: rgba(255,255,255,0.1) !important;
        color: var(--text-dim) !important;
        font-size: 9px;
        padding: 1px 6px;
        border-radius: 4px;
    }
    .tool-badge {
        background: rgba(249, 115, 22, 0.2) !important;
        color: var(--tool) !important;
        font-size: 9px;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .chat-content {
        font-size: 13px;
        line-height: 1.6;
        color: var(--text-secondary) !important;
        background: transparent !important;
    }
    
    /* Log Items */
    .log-item {
        font-size: 11px;
        padding: 6px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .log-item.success { color: var(--success) !important; background: rgba(16, 185, 129, 0.1) !important; }
    .log-item.warning { color: var(--warning) !important; background: rgba(245, 158, 11, 0.1) !important; }
    .log-item.error { color: var(--error) !important; background: rgba(239, 68, 68, 0.1) !important; }
    .log-item.info { color: var(--primary) !important; background: transparent !important; }
    .log-item.dim { color: var(--text-dim) !important; background: transparent !important; }
    .log-item.tool { color: var(--tool) !important; background: transparent !important; }

    /* File Tree */
    .file-tree {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        background: var(--bg-input) !important;
        padding: 15px;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid var(--border-color);
    }
    .tree-item { padding: 3px 0; color: var(--text-secondary) !important; background: transparent !important; }
    .tree-item.file { color: var(--success) !important; }
    .tree-item.dir { color: var(--primary) !important; font-weight: bold; }
    .indent { display: inline-block; width: 15px; }
    
    /* Markdown Styles */
    .markdown-text h1, .markdown-text h2, .markdown-text h3 { color: var(--text-primary) !important; }
    .markdown-text p { color: var(--text-secondary) !important; }
    .markdown-text strong { color: var(--text-primary) !important; }
    blockquote { border-left-color: var(--primary) !important; background: var(--bg-input) !important; color: var(--text-dim) !important; }
    """
    
    initial_projects = get_projects()
    
    with gr.Blocks(title="CODE EVAL") as app:
        # Inject CSS directly via HTML
        gr.HTML(f"<style>{css}</style>")
        
        # Header
        gr.HTML('''
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 20px 5px; margin-bottom: 20px; border-bottom: 1px solid var(--border-color);">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, var(--primary), var(--success)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 20px; color: white;">CE</div>
                <div>
                    <div style="font-size: 22px; font-weight: 700; letter-spacing: -0.5px; color: white;">Code Eval</div>
                    <div style="font-size: 12px; color: var(--text-secondary);">AI-Powered Code Analysis System</div>
                </div>
            </div>
            <div style="display: flex; gap: 10px;">
                <div style="padding: 6px 12px; background: rgba(16, 185, 129, 0.1); color: var(--success); border-radius: 20px; font-size: 12px; font-weight: 600;">v3.0 Stable</div>
            </div>
        </div>
        ''')
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: SINGLE FILE
            # ============================================================
            with gr.TabItem("Single File"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠️ Configuration")
                        sf_project = gr.Dropdown(label="Project", choices=initial_projects, value=initial_projects[0] if initial_projects else None)
                        sf_file = gr.Dropdown(label="Target File", choices=get_files_in_project(initial_projects[0]) if initial_projects else [])
                        sf_provider = gr.Dropdown(label="Model", choices=list(MODEL_MAPPINGS.keys()), value=DEFAULT_PROVIDER)
                        sf_auto_fix = gr.Checkbox(label="Auto-Fix Loop", value=True)
                        
                        with gr.Row():
                            sf_run_btn = gr.Button("Analyze & Fix", variant="primary")
                            sf_refresh = gr.Button("Refresh")
                        
                        gr.Markdown("### 📊 Status")
                        sf_status = gr.HTML(value=create_status("idle", "Ready", "Waiting for input..."))
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 💻 Editor & Logs")
                        with gr.Tabs():
                            with gr.TabItem("Live Logs"):
                                sf_logs = gr.HTML(value=create_logs([]))
                            with gr.TabItem("Code Editor"):
                                sf_code = gr.Code(label="", language="python", lines=25, interactive=True)
                                sf_save_btn = gr.Button("Save Changes")
                                sf_save_status = gr.HTML()

            # ============================================================
            # TAB 2: QUICK SCAN
            # ============================================================
            with gr.TabItem("Quick Scan"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Scanner Config")
                        qs_project = gr.Dropdown(label="Project", choices=initial_projects, value=initial_projects[0] if initial_projects else None)
                        with gr.Row():
                            qs_scan_btn = gr.Button("Start Scan", variant="primary")
                            qs_refresh = gr.Button("Refresh")
                        
                        qs_status = gr.HTML(value=create_status("idle", "Ready", "Select project to scan"))
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📈 Scan Results")
                        qs_details = gr.HTML()
                        qs_tree = gr.HTML()

            # ============================================================
            # TAB 3: MULTI-AGENT
            # ============================================================
            with gr.TabItem("Multi-Agent Workflow"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 Workflow Config")
                        ma_project = gr.Dropdown(label="Project", choices=initial_projects, value=initial_projects[0] if initial_projects else None)
                        ma_provider = gr.Dropdown(label="Model", choices=list(MODEL_MAPPINGS.keys()), value=DEFAULT_PROVIDER)
                        
                        with gr.Row():
                            ma_run_btn = gr.Button("Start Analysis", variant="primary")
                            ma_refresh = gr.Button("Refresh")
                            
                        gr.Markdown("### 📡 Live Status")
                        ma_status = gr.HTML(value=create_status("idle", "Ready", "Workflow idle"))
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Agent Interaction")
                        ma_logs = gr.HTML(value=create_logs([]))
                        ma_report = gr.HTML()

            # ============================================================
            # TAB 4: HELP
            # ============================================================
            with gr.TabItem("Documentation"):
                gr.Markdown("""
                ### 📘 How to Use Code Eval
                
                **1. Single File Mode**
                > Great for fixing specific bugs in one file. It uses a 3-agent loop: Analyzer -> Executor -> Fixer.
                
                **2. Quick Scan**
                > Runs a static analysis (AST) to check for syntax errors and file structure. No AI cost.
                
                **3. Multi-Agent Workflow**
                > The powerhouse. Orchestrates 5 agents (Scanner, Analyzer, Tester, Fixer, Reporter) to solve complex repo-level issues.
                """)
    
        # Event Wiring
        def update_file_list(project):
            files = get_files_in_project(project)
            return gr.update(choices=files, value=files[0] if files else None)
        
        def refresh_projects():
            projects = get_projects()
            return gr.update(choices=projects, value=projects[0] if projects else None)
        
        sf_project.change(fn=update_file_list, inputs=[sf_project], outputs=[sf_file])
        sf_file.change(fn=load_file_content, inputs=[sf_project, sf_file], outputs=[sf_code])
        sf_run_btn.click(fn=run_single_file_analysis, inputs=[sf_project, sf_file, sf_provider, sf_auto_fix], outputs=[sf_status, sf_logs, sf_code])
        sf_save_btn.click(fn=save_file, inputs=[sf_project, sf_file, sf_code], outputs=[sf_save_status])
        sf_refresh.click(fn=refresh_projects, outputs=[sf_project])
        
        qs_scan_btn.click(fn=run_quick_scan, inputs=[qs_project], outputs=[qs_status, qs_details, qs_tree])
        qs_refresh.click(fn=refresh_projects, outputs=[qs_project])
        
        ma_run_btn.click(fn=run_repo_analysis, inputs=[ma_project, ma_provider], outputs=[ma_status, ma_logs, ma_report])
        ma_refresh.click(fn=refresh_projects, outputs=[ma_project])
    
    return app


if __name__ == "__main__":
    BENCH_DIR.mkdir(exist_ok=True)
    app = create_ui()
    app.launch(server_name="127.0.0.1", share=False, inbrowser=True)
