#!/usr/bin/env python
"""
Code Eval v4.0 - NiceGUI Modern Application
Advanced Multi-Agent Code Analysis System with Next-Gen UI
"""
import sys
import asyncio
import uuid
import traceback
import queue
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicegui import ui, app
from nicegui.events import ValueChangeEventArguments

# Import theme and components
from .theme import COLORS, GRADIENTS, GLOBAL_CSS, QUASAR_CONFIG
from .components import (
    StatusIndicator, MetricCard, ChatMessage, FileTree,
    LogViewer, WorkflowVisualizer, CodeDiffViewer, AIPromptInput
)

# Import workflow components
from workflow import create_workflow, create_multi_agent_workflow
from agent import quick_scan
from config import DEFAULT_PROVIDER, MODEL_MAPPINGS


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.bench_dir = Path(__file__).parent.parent / "code_bench"
        self.current_project: Optional[str] = None
        self.current_file: Optional[str] = None
        self.original_code: str = ""
        self.modified_code: str = ""
        self.is_running: bool = False
        self.workflow = None
        self.repo_workflow = None
        self.conversation_history: List[Dict[str, Any]] = []

state = AppState()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_projects() -> List[str]:
    """Get available projects in code_bench"""
    state.bench_dir.mkdir(exist_ok=True)
    projects = []
    for d in state.bench_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__':
            projects.append(d.name)
    return sorted(projects)


def get_files_in_project(project_name: str) -> List[str]:
    """Get Python files in a project"""
    if not project_name:
        return []
    path = state.bench_dir / project_name
    if not path.exists():
        return []
    
    files = []
    for f in path.rglob("*.py"):
        if "__pycache__" not in str(f) and not f.name.startswith('.'):
            files.append(str(f.relative_to(path)))
    return sorted(files)


def load_file_content(project: str, file_path: str) -> str:
    """Load file content"""
    if not project or not file_path:
        return ""
    full_path = state.bench_dir / project / file_path
    if not full_path.exists():
        return f"# File not found: {file_path}"
    try:
        return full_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"# Error reading file: {str(e)}"


def save_file_content(project: str, file_path: str, content: str) -> bool:
    """Save file content"""
    if not project or not file_path:
        return False
    full_path = state.bench_dir / project / file_path
    try:
        full_path.write_text(content, encoding='utf-8')
        return True
    except Exception:
        return False


# ============================================================================
# UI PAGES
# ============================================================================

def create_header():
    """Create the app header"""
    with ui.header().classes('items-center justify-between px-6').style(
        f'background: {COLORS["bg_secondary"]}; border-bottom: 1px solid {COLORS["border"]}'
    ):
        with ui.element('div').classes('flex items-center gap-4'):
            # Logo
            with ui.element('div').classes('flex items-center gap-3'):
                with ui.element('div').style(
                    f'background: {GRADIENTS["primary"]}; padding: 8px 12px; border-radius: 10px;'
                ):
                    ui.label('CE').classes('cyber-heading text-xl font-bold').style('color: white')
                
                with ui.element('div'):
                    ui.label('Code Eval').classes('cyber-heading text-lg').style(f'color: {COLORS["text_primary"]}')
                    ui.label('AI-Powered Analysis').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
        
        with ui.element('div').classes('flex items-center gap-3'):
            ui.badge('v4.0', color='green').props('outline')
            ui.badge('NiceGUI', color='blue').props('outline')


def create_sidebar(project_select, file_select, log_viewer, status_indicator):
    """Create the sidebar with project/file selection"""
    with ui.left_drawer(bordered=True).classes('p-4').style(
        f'background: {COLORS["bg_secondary"]}; border-right: 1px solid {COLORS["border"]};'
    ):
        # Project Selection
        ui.label('Project').classes('text-sm font-bold mb-2').style(f'color: {COLORS["text_dim"]}')
        
        projects = get_projects()
        project_select.options = projects
        if projects:
            project_select.value = projects[0]
            state.current_project = projects[0]
        
        ui.button('Refresh', on_click=lambda: refresh_projects(project_select)).props('flat dense').classes('mb-4')
        
        # File Selection
        ui.label('Target File').classes('text-sm font-bold mb-2 mt-4').style(f'color: {COLORS["text_dim"]}')
        
        if state.current_project:
            files = get_files_in_project(state.current_project)
            file_select.options = files
            if files:
                file_select.value = files[0]
                state.current_file = files[0]
        
        # Provider Selection
        ui.label('LLM Provider').classes('text-sm font-bold mb-2 mt-4').style(f'color: {COLORS["text_dim"]}')
        provider_select = ui.select(
            list(MODEL_MAPPINGS.keys()),
            value=DEFAULT_PROVIDER
        ).classes('w-full')
        
        ui.separator().classes('my-4')
        
        # Status
        StatusIndicator("idle", "Ready", "Select a file to begin")


def refresh_projects(project_select):
    """Refresh project list"""
    projects = get_projects()
    project_select.options = projects
    ui.notify('Projects refreshed', color='positive')


@ui.page('/')
def main_page():
    """Main application page"""
    # Add custom CSS
    ui.add_head_html(f'<style>{GLOBAL_CSS}</style>')
    
    # Apply dark theme
    ui.dark_mode(True)
    
    # Header
    create_header()
    
    # Main content with tabs
    with ui.element('div').classes('w-full p-6').style(f'background: {COLORS["bg_primary"]}; min-height: 100vh; width: 100%;'):
        
        with ui.tabs().classes('w-full') as tabs:
            single_tab = ui.tab('single', label='Single File', icon='description')
            scan_tab = ui.tab('scan', label='Quick Scan', icon='search')
            multi_tab = ui.tab('multi', label='Multi-Agent', icon='psychology')
            chat_tab = ui.tab('chat', label='AI Chat', icon='chat')
            settings_tab = ui.tab('settings', label='Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=single_tab).classes('w-full'):
            
            # ============================================================
            # SINGLE FILE TAB
            # ============================================================
            with ui.tab_panel(single_tab):
                create_single_file_panel()
            
            # ============================================================
            # QUICK SCAN TAB
            # ============================================================
            with ui.tab_panel(scan_tab):
                create_quick_scan_panel()
            
            # ============================================================
            # MULTI-AGENT TAB
            # ============================================================
            with ui.tab_panel(multi_tab):
                create_multi_agent_panel()
            
            # ============================================================
            # AI CHAT TAB
            # ============================================================
            with ui.tab_panel(chat_tab):
                create_ai_chat_panel()
            
            # ============================================================
            # SETTINGS TAB
            # ============================================================
            with ui.tab_panel(settings_tab):
                create_settings_panel()


def create_single_file_panel():
    """Create single file analysis panel"""
    with ui.element('div').classes('w-full flex gap-6'):
        # Left column - Configuration
        with ui.element('div').classes('w-80'):
            with ui.element('div').classes('glass-card p-4 mb-4'):
                ui.label('Configuration').classes('text-lg font-bold mb-4').style(f'color: {COLORS["primary"]}')
                
                # Project select
                ui.label('Project').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                project_select = ui.select(
                    get_projects(),
                    value=get_projects()[0] if get_projects() else None,
                    on_change=lambda e: on_project_change(e, file_select, code_editor)
                ).classes('w-full mb-3')
                
                # File select
                ui.label('File').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                initial_files = get_files_in_project(project_select.value) if project_select.value else []
                file_select = ui.select(
                    initial_files,
                    value=initial_files[0] if initial_files else None,
                    on_change=lambda e: on_file_change(e, project_select, code_editor)
                ).classes('w-full mb-3')
                
                # Provider select
                ui.label('Model').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                provider_select = ui.select(
                    list(MODEL_MAPPINGS.keys()),
                    value=DEFAULT_PROVIDER
                ).classes('w-full mb-3')
                
                # Auto-fix toggle
                auto_fix = ui.switch('Auto-Fix Loop', value=True).classes('mb-4')
                
                # Action buttons
                with ui.element('div').classes('flex gap-2'):
                    analyze_btn = ui.button('Analyze & Fix', on_click=lambda: run_single_analysis(
                        project_select.value, file_select.value, provider_select.value, 
                        auto_fix.value, status_display, log_viewer, code_editor
                    )).props('color=primary').classes('flex-grow btn-primary')
                    
                    ui.button(icon='refresh', on_click=lambda: refresh_projects_ui(project_select)).props('flat')
            
            # Status display
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Status').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                status_display = ui.element('div')
                with status_display:
                    StatusIndicator("idle", "Ready", "Waiting for input...")
        
        # Right column - Editor and Logs
        with ui.element('div').classes('flex-grow'):
            with ui.tabs().classes('w-full') as inner_tabs:
                log_tab = ui.tab('logs', label='Live Logs', icon='terminal')
                editor_tab = ui.tab('editor', label='Code Editor', icon='code')
                diff_tab = ui.tab('diff', label='Diff View', icon='compare')
            
            with ui.tab_panels(inner_tabs, value=log_tab).classes('w-full'):
                with ui.tab_panel(log_tab):
                    log_viewer = LogViewer()
                
                with ui.tab_panel(editor_tab):
                    with ui.element('div').classes('glass-card p-4'):
                        code_editor = ui.codemirror(
                            value=load_file_content(project_select.value, file_select.value) if project_select.value and file_select.value else "# Select a file",
                            language='python',
                            theme='oneDark'
                        ).classes('w-full').style('min-height: 500px; font-size: 14px;')
                        
                        with ui.element('div').classes('flex justify-between mt-3'):
                            ui.button('Save Changes', on_click=lambda: save_code(
                                project_select.value, file_select.value, code_editor.value
                            )).props('color=positive')
                            ui.button('Reset', on_click=lambda: reset_code(
                                project_select.value, file_select.value, code_editor
                            )).props('flat')
                
                with ui.tab_panel(diff_tab):
                    ui.label('Code comparison will appear here after analysis').style(f'color: {COLORS["text_dim"]}')


def create_quick_scan_panel():
    """Create quick scan panel"""
    with ui.element('div').classes('w-full flex gap-6'):
        # Left column
        with ui.element('div').classes('w-80'):
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Quick Scan').classes('text-lg font-bold mb-4').style(f'color: {COLORS["scanner"]}')
                ui.label('Fast local analysis without LLM calls').classes('text-sm mb-4').style(f'color: {COLORS["text_dim"]}')
                
                # Project select
                ui.label('Project').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                scan_project = ui.select(
                    get_projects(),
                    value=get_projects()[0] if get_projects() else None
                ).classes('w-full mb-4')
                
                ui.button('Start Scan', on_click=lambda: run_quick_scan(
                    scan_project.value, scan_result, scan_tree
                )).props('color=primary').classes('w-full btn-primary')
        
        # Right column - Results
        with ui.element('div').classes('flex-grow'):
            with ui.element('div').classes('glass-card p-4 mb-4'):
                ui.label('Scan Results').classes('text-lg font-bold mb-4').style(f'color: {COLORS["primary"]}')
                scan_result = ui.element('div')
                with scan_result:
                    ui.label('Run a scan to see results').style(f'color: {COLORS["text_dim"]}')
            
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Project Structure').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                scan_tree = ui.element('div')


def create_multi_agent_panel():
    """Create multi-agent workflow panel"""
    with ui.element('div').classes('w-full flex gap-6'):
        # Left column
        with ui.element('div').classes('w-80'):
            with ui.element('div').classes('glass-card p-4 mb-4'):
                ui.label('Autonomous Agent System').classes('text-lg font-bold mb-2').style(f'color: {COLORS["primary"]}')
                ui.label('Supervisor-driven autonomous analysis').classes('text-sm mb-4').style(f'color: {COLORS["text_dim"]}')
                
                # Project select
                ui.label('Project').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                ma_project = ui.select(
                    get_projects(),
                    value=get_projects()[0] if get_projects() else None
                ).classes('w-full mb-3')
                
                # Provider select
                ui.label('Model').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                ma_provider = ui.select(
                    list(MODEL_MAPPINGS.keys()),
                    value=DEFAULT_PROVIDER
                ).classes('w-full mb-4')
                
                ui.button('Start Analysis', on_click=lambda: run_multi_agent(
                    ma_project.value, ma_provider.value, ma_status, ma_logs, ma_workflow
                )).props('color=primary').classes('w-full btn-primary')
            
            # Status
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Status').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                ma_status = ui.element('div')
                with ma_status:
                    StatusIndicator("idle", "Ready", "Workflow idle")
        
        # Right column
        with ui.element('div').classes('flex-grow'):
            # Workflow visualizer
            ma_workflow = ui.element('div').classes('mb-4')
            with ma_workflow:
                WorkflowVisualizer()
            
            # Logs
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Agent Interactions').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                ma_logs = LogViewer()


def create_ai_chat_panel():
    """Create AI chat panel for interactive code assistance"""
    with ui.element('div').classes('w-full flex gap-6 h-full'):
        # Chat area
        with ui.element('div').classes('flex-grow flex flex-col'):
            with ui.element('div').classes('glass-card p-4 flex-grow').style('min-height: 500px; display: flex; flex-direction: column;'):
                ui.label('AI Code Assistant').classes('text-lg font-bold mb-4').style(f'color: {COLORS["llm"]}')
                
                # Chat messages container
                chat_container = ui.scroll_area().classes('flex-grow').style(
                    f'background: {COLORS["bg_secondary"]}; border-radius: 8px; padding: 16px;'
                )
                with chat_container:
                    ChatMessage("system", "Hello! I'm your AI code assistant. Ask me anything about your code, and I can help you analyze, fix, or improve it.")
                
                # Input area
                with ui.element('div').classes('mt-4'):
                    with ui.element('div').classes('flex gap-2'):
                        chat_input = ui.textarea(
                            placeholder='Ask me anything about your code...'
                        ).props('outlined dense rows=2').classes('flex-grow')
                        
                        ui.button(icon='send', on_click=lambda: send_chat_message(
                            chat_input, chat_container
                        )).props('round color=primary').classes('self-end')
            
            # Quick actions
            with ui.element('div').classes('glass-card p-4 mt-4'):
                ui.label('Quick Actions').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                
                with ui.element('div').classes('flex flex-wrap gap-2'):
                    quick_prompts = [
                        ("🔍 Find bugs", "Find potential bugs in the current code"),
                        ("⚡ Optimize", "Suggest performance optimizations"),
                        ("📝 Document", "Add documentation and type hints"),
                        ("🧪 Test", "Generate unit tests for this code"),
                        ("🔒 Security", "Check for security vulnerabilities"),
                        ("♻️ Refactor", "Suggest refactoring improvements"),
                    ]
                    
                    for label, prompt in quick_prompts:
                        ui.button(label, on_click=lambda e, p=prompt: quick_action(p, chat_input)).props('outline dense').classes('text-xs')
        
        # Context sidebar
        with ui.element('div').classes('w-80'):
            with ui.element('div').classes('glass-card p-4'):
                ui.label('Context').classes('text-sm font-bold mb-3').style(f'color: {COLORS["text_dim"]}')
                
                # Current file info
                ui.label('Current File:').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
                ui.label(state.current_file or 'No file selected').classes('text-sm mb-3').style(f'color: {COLORS["text_secondary"]}')
                
                # File content preview
                ui.label('Code Preview:').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
                with ui.element('div').classes('mono text-xs p-2 rounded mt-1').style(
                    f'background: {COLORS["code_bg"]}; max-height: 300px; overflow: auto;'
                ):
                    preview = state.original_code[:500] + "..." if len(state.original_code) > 500 else state.original_code
                    ui.label(preview or "No code loaded").style(f'color: {COLORS["text_secondary"]}; white-space: pre-wrap;')


def create_settings_panel():
    """Create settings panel"""
    with ui.element('div').classes('w-full max-w-4xl mx-auto'):
        with ui.element('div').classes('glass-card p-6'):
            ui.label('Settings').classes('text-2xl font-bold mb-6').style(f'color: {COLORS["primary"]}')
            
            # LLM Settings
            with ui.element('div').classes('mb-6'):
                ui.label('LLM Configuration').classes('text-lg font-semibold mb-4').style(f'color: {COLORS["text_primary"]}')
                
                with ui.element('div').classes('grid grid-cols-2 gap-4'):
                    with ui.element('div'):
                        ui.label('Default Provider').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                        ui.select(list(MODEL_MAPPINGS.keys()), value=DEFAULT_PROVIDER).classes('w-full')
                    
                    with ui.element('div'):
                        ui.label('Max Attempts').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                        ui.number(value=5, min=1, max=10).classes('w-full')
            
            ui.separator()
            
            # Execution Settings
            with ui.element('div').classes('mb-6 mt-6'):
                ui.label('Execution Settings').classes('text-lg font-semibold mb-4').style(f'color: {COLORS["text_primary"]}')
                
                with ui.element('div').classes('grid grid-cols-2 gap-4'):
                    with ui.element('div'):
                        ui.label('Timeout (seconds)').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                        ui.number(value=30, min=5, max=300).classes('w-full')
                    
                    with ui.element('div'):
                        ui.label('Working Directory').classes('text-sm mb-1').style(f'color: {COLORS["text_dim"]}')
                        ui.input(value='code_bench').classes('w-full')
            
            ui.separator()
            
            # Theme Settings
            with ui.element('div').classes('mt-6'):
                ui.label('Appearance').classes('text-lg font-semibold mb-4').style(f'color: {COLORS["text_primary"]}')
                
                ui.switch('Dark Mode', value=True)
                ui.switch('Show Line Numbers', value=True)
                ui.switch('Auto-scroll Logs', value=True)


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def on_project_change(e, file_select, code_editor):
    """Handle project selection change"""
    project = e.value
    state.current_project = project
    files = get_files_in_project(project)
    file_select.options = files
    if files:
        file_select.value = files[0]
        state.current_file = files[0]
        content = load_file_content(project, files[0])
        state.original_code = content
        code_editor.value = content


def on_file_change(e, project_select, code_editor):
    """Handle file selection change"""
    file_path = e.value
    state.current_file = file_path
    content = load_file_content(project_select.value, file_path)
    state.original_code = content
    code_editor.value = content


def refresh_projects_ui(project_select):
    """Refresh projects in UI"""
    projects = get_projects()
    project_select.options = projects
    ui.notify('Projects refreshed', color='positive')


def save_code(project: str, file_path: str, content: str):
    """Save code to file"""
    if save_file_content(project, file_path, content):
        ui.notify('File saved successfully', color='positive')
    else:
        ui.notify('Failed to save file', color='negative')


def reset_code(project: str, file_path: str, code_editor):
    """Reset code to original"""
    content = load_file_content(project, file_path)
    code_editor.value = content
    ui.notify('Code reset to original', color='info')


async def run_single_analysis(project: str, file_path: str, provider: str, auto_fix: bool, 
                               status_display, log_viewer, code_editor):
    """Run single file analysis"""
    if not project or not file_path:
        ui.notify('Please select a project and file', color='warning')
        return
    
    if state.is_running:
        ui.notify('Analysis already running', color='warning')
        return
    
    state.is_running = True
    
    # Update status
    status_display.clear()
    with status_display:
        StatusIndicator("running", "Analyzing", f"Processing {file_path}...")
    
    log_viewer.clear()
    log_viewer.add_log(f"Starting analysis on {file_path}", "info")
    log_viewer.add_log(f"Provider: {provider}, Auto-fix: {auto_fix}", "info")
    
    try:
        full_path = state.bench_dir / project / file_path
        original_code = full_path.read_text(encoding='utf-8')
        state.original_code = original_code
        
        # Create workflow
        log_viewer.add_log("Creating code improvement workflow...", "info")
        workflow = create_workflow(
            llm_provider=provider,
            max_attempts=5 if auto_fix else 1
        )
        
        thread_id = str(uuid.uuid4())
        log_viewer.add_log(f"Thread: {thread_id[:8]}...", "info")
        
        # Use queue to communicate between thread and async
        update_queue = queue.Queue()
        
        def run_workflow_sync():
            """Run workflow in a separate thread"""
            try:
                for update in workflow.stream_run(str(full_path), original_code, thread_id=thread_id):
                    update_queue.put(('update', update))
                update_queue.put(('done', None))
            except Exception as e:
                update_queue.put(('error', str(e)))
        
        # Start workflow in background thread
        workflow_thread = threading.Thread(target=run_workflow_sync, daemon=True)
        workflow_thread.start()
        
        # Process updates from queue
        step = 0
        current_node = None
        final_result = None
        
        while True:
            await asyncio.sleep(0.1)  # Keep event loop responsive
            
            try:
                while True:
                    msg_type, data = update_queue.get_nowait()
                    
                    if msg_type == 'done':
                        # Process final results
                        if final_result:
                            success = final_result.get("execution_success", False)
                            attempts = final_result.get("execution_attempts", 0)
                            
                            status_display.clear()
                            with status_display:
                                if success:
                                    StatusIndicator("success", "Complete", f"SUCCESS after {attempts} attempt(s)")
                                    log_viewer.add_log(f"SUCCESS after {attempts} attempt(s)", "success")
                                else:
                                    StatusIndicator("warning", "Complete", f"INCOMPLETE after {attempts} attempt(s)")
                                    log_viewer.add_log(f"INCOMPLETE after {attempts} attempt(s)", "warning")
                            
                            # Update code editor with final code
                            try:
                                final_code = full_path.read_text(encoding='utf-8')
                                state.modified_code = final_code
                                code_editor.value = final_code
                            except:
                                pass
                        state.is_running = False
                        return
                    
                    elif msg_type == 'error':
                        raise Exception(data)
                    
                    elif msg_type == 'update':
                        update = data
                        step += 1
                        
                        if update:
                            node_name = list(update.keys())[0]
                            node_data = update[node_name]
                            
                            if node_name != current_node:
                                agent_name = node_name.replace("_node", "").upper()
                                log_viewer.add_chat("system", f"Step {step}: {agent_name}")
                                current_node = node_name
                            
                            if isinstance(node_data, dict):
                                if node_data.get("code_analysis"):
                                    analysis = node_data["code_analysis"][:400]
                                    log_viewer.add_chat("analyzer", analysis, step)
                                
                                if "execution_success" in node_data:
                                    success = node_data["execution_success"]
                                    if success:
                                        log_viewer.add_log("Code executed successfully", "success")
                                    else:
                                        error = node_data.get("last_error", "Unknown error")[:200]
                                        log_viewer.add_chat("executor", f"Execution failed: {error}", step)
                                
                                if node_data.get("modification_history"):
                                    log_viewer.add_chat("fixer", "Applied code modification", step, True)
                                
                                final_result = node_data
                                
            except queue.Empty:
                if not workflow_thread.is_alive() and update_queue.empty():
                    break
        
    except Exception as e:
        log_viewer.add_log(f"Error: {str(e)}", "error")
        status_display.clear()
        with status_display:
            StatusIndicator("error", "Failed", str(e)[:50])
    finally:
        state.is_running = False


def run_quick_scan(project: str, result_container, tree_container):
    """Run quick scan on project"""
    if not project:
        ui.notify('Please select a project', color='warning')
        return
    
    project_path = state.bench_dir / project
    
    try:
        result = quick_scan(str(project_path))
        
        result_container.clear()
        with result_container:
            # Metrics
            with ui.element('div').classes('grid grid-cols-3 gap-4 mb-4'):
                MetricCard(result['total_files'], 'PYTHON FILES', COLORS["success"], 'description')
                MetricCard(result['files_with_syntax_errors'], 'SYNTAX ERRORS', 
                          COLORS["error"] if result['files_with_syntax_errors'] > 0 else COLORS["success"], 'error')
                MetricCard('YES' if result['has_tests'] else 'NO', 'HAS TESTS', COLORS["info"], 'science')
            
            # Errors list
            if result.get('syntax_errors'):
                ui.label('Syntax Errors:').classes('text-sm font-bold mb-2').style(f'color: {COLORS["error"]}')
                for err in result['syntax_errors'][:5]:
                    with ui.element('div').classes('log-item log-error mb-2'):
                        ui.label(f"{err['file']}: {'; '.join(err['errors'][:2])}")
            else:
                ui.label('✓ All files have valid syntax!').style(f'color: {COLORS["success"]}')
        
        # File tree
        tree_container.clear()
        with tree_container:
            FileTree(project_path)
        
        ui.notify('Scan complete', color='positive')
        
    except Exception as e:
        ui.notify(f'Scan failed: {str(e)}', color='negative')


async def run_multi_agent(project: str, provider: str, status_container, log_viewer, workflow_container):
    """Run multi-agent workflow"""
    if not project:
        ui.notify('Please select a project', color='warning')
        return
    
    if state.is_running:
        ui.notify('Analysis already running', color='warning')
        return
    
    state.is_running = True
    project_path = state.bench_dir / project
    
    # Reset workflow visualizer
    workflow_container.clear()
    with workflow_container:
        wf_viz = WorkflowVisualizer()
    
    status_container.clear()
    with status_container:
        StatusIndicator("running", "Initializing", "Creating workflow...")
    
    log_viewer.clear()
    log_viewer.add_log(f"Starting multi-agent analysis on {project}", "info")
    
    try:
        repo_workflow = create_multi_agent_workflow(provider, max_fix_attempts=5)
        
        step_count = 0
        current_agent = None
        
        # Use queue to communicate between thread and async
        update_queue = queue.Queue()
        
        def run_workflow_sync():
            """Run workflow in a separate thread"""
            try:
                for update in repo_workflow.stream_run(str(project_path)):
                    update_queue.put(('update', update))
                update_queue.put(('done', None))
            except Exception as e:
                update_queue.put(('error', str(e)))
        
        # Start workflow in background thread
        workflow_thread = threading.Thread(target=run_workflow_sync, daemon=True)
        workflow_thread.start()
        
        # Process updates from queue
        while True:
            await asyncio.sleep(0.1)  # Keep event loop responsive
            
            # Process all available updates
            try:
                while True:
                    msg_type, data = update_queue.get_nowait()
                    
                    if msg_type == 'done':
                        # Complete
                        wf_viz.complete(current_agent)
                        status_container.clear()
                        with status_container:
                            StatusIndicator("success", "Complete", f"Finished in {step_count} steps")
                        log_viewer.add_log(f"Workflow complete in {step_count} steps", "success")
                        ui.notify('Multi-agent analysis complete', color='positive')
                        state.is_running = False
                        return
                    
                    elif msg_type == 'error':
                        raise Exception(data)
                    
                    elif msg_type == 'update':
                        update = data
                        step_count += 1
                        
                        if update:
                            node = list(update.keys())[0]
                            node_data = update.get(node, {})
                            
                            # Update workflow visualizer
                            if node != current_agent:
                                current_agent = node
                                wf_viz.set_active(node)
                                log_viewer.add_chat("system", f"Step {step_count}: {node.upper()}")
                            
                            # Process step logs
                            if isinstance(node_data, dict):
                                step_logs = node_data.get("step_logs", [])
                                for sl in step_logs[-3:]:  # Show last 3 logs
                                    agent = sl.get("agent", "")
                                    turn = sl.get("turn", 0)
                                    log_type = sl.get("type", "")
                                    
                                    if log_type == "llm_response":
                                        content = sl.get("content", "")[:300]
                                        has_tools = sl.get("has_tool_calls", False)
                                        log_viewer.add_chat(agent.lower(), content, turn, has_tools)
                                    elif log_type == "tool_call":
                                        tool = sl.get("tool", "unknown")
                                        result = sl.get("result", "")[:150]
                                        log_viewer.add_chat("tool", f"{tool}: {result}")
                        
                        if step_count > 30:
                            log_viewer.add_log("Reached iteration limit", "warning")
                            state.is_running = False
                            return
                            
            except queue.Empty:
                # No more updates available, continue waiting
                if not workflow_thread.is_alive() and update_queue.empty():
                    break
        
    except Exception as e:
        log_viewer.add_log(f"Error: {str(e)}", "error")
        status_container.clear()
        with status_container:
            StatusIndicator("error", "Failed", str(e)[:50])
    finally:
        state.is_running = False


def send_chat_message(chat_input, chat_container):
    """Send a chat message"""
    message = chat_input.value
    if not message:
        return
    
    # Add user message
    with chat_container:
        ChatMessage("user", message)
    
    # Simulate AI response (in production, this would call the LLM)
    with chat_container:
        ChatMessage("llm", f"I'll analyze your request: '{message[:50]}...'\n\nThis feature is coming soon! In the full implementation, I would use the configured LLM to provide intelligent responses about your code.", turn=1)
    
    chat_input.value = ""
    chat_container.scroll_to(percent=1)


def quick_action(prompt: str, chat_input):
    """Fill chat input with quick action prompt"""
    chat_input.value = prompt


# ============================================================================
# APP ENTRY POINT
# ============================================================================

def run_app(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    """Run the NiceGUI application"""
    
    # Ensure bench directory exists
    state.bench_dir.mkdir(exist_ok=True)
    
    # Configure app
    app.native.settings['ALLOW_DOWNLOADS'] = True
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🚀 CODE EVAL v4.0 - NiceGUI Edition                       ║
    ║                                                              ║
    ║   Starting server at: http://{host}:{port}                   ║
    ║                                                              ║
    ║   Features:                                                  ║
    ║   • Single File Analysis with Auto-Fix                      ║
    ║   • Quick Scan (No LLM)                                     ║
    ║   • Multi-Agent Workflow with Visualization                 ║
    ║   • AI Chat Assistant                                       ║
    ║   • Modern Cyberpunk UI Theme                               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    ui.run(
        host=host,
        port=port,
        title="Code Eval v4.0",
        favicon="🚀",
        dark=True,
        reload=reload,
        show=True
    )


if __name__ == "__main__":
    run_app()

