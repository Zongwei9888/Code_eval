"""
Reusable UI Components for Code Eval v4.0
"""
from nicegui import ui
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .theme import COLORS, GRADIENTS


class StatusIndicator:
    """Animated status indicator component"""
    
    STATUS_MAP = {
        "success": ("✓", COLORS["success"], "SUCCESS"),
        "error": ("✗", COLORS["error"], "ERROR"),
        "warning": ("⚠", COLORS["warning"], "WARNING"),
        "running": ("◉", COLORS["primary"], "RUNNING"),
        "idle": ("○", COLORS["text_dim"], "IDLE"),
        "pending": ("◎", COLORS["text_secondary"], "PENDING"),
    }
    
    def __init__(self, status: str = "idle", title: str = "", subtitle: str = ""):
        self.container = None
        self.status = status
        self.title = title
        self.subtitle = subtitle
        self._build()
    
    def _build(self):
        icon, color, label = self.STATUS_MAP.get(self.status, self.STATUS_MAP["idle"])
        pulse_class = "pulse-glow" if self.status == "running" else ""
        
        with ui.element('div').classes(f'glass-card p-4 flex items-center gap-4 {pulse_class}') as self.container:
            self.container.style(f'border-left: 4px solid {color}')
            
            with ui.element('div').classes('text-3xl font-bold').style(f'color: {color}'):
                ui.label(icon)
            
            with ui.element('div'):
                ui.label(self.title or label).classes('text-lg font-semibold')
                if self.subtitle:
                    ui.label(self.subtitle).classes('text-sm').style(f'color: {COLORS["text_secondary"]}')
    
    def update(self, status: str, title: str = "", subtitle: str = ""):
        """Update status indicator"""
        self.status = status
        self.title = title
        self.subtitle = subtitle
        if self.container:
            self.container.clear()
            with self.container:
                icon, color, label = self.STATUS_MAP.get(status, self.STATUS_MAP["idle"])
                pulse_class = "pulse-glow" if status == "running" else ""
                self.container.classes(f'glass-card p-4 flex items-center gap-4 {pulse_class}')
                self.container.style(f'border-left: 4px solid {color}')
                
                with ui.element('div').classes('text-3xl font-bold').style(f'color: {color}'):
                    ui.label(icon)
                
                with ui.element('div'):
                    ui.label(title or label).classes('text-lg font-semibold')
                    if subtitle:
                        ui.label(subtitle).classes('text-sm').style(f'color: {COLORS["text_secondary"]}')


class MetricCard:
    """Metric display card with animated value"""
    
    def __init__(self, value: Any, label: str, color: str = None, icon: str = None):
        self.value = value
        self.label = label
        self.color = color or COLORS["primary"]
        self.icon = icon
        self.value_label = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('metric-card hover-lift'):
            if self.icon:
                ui.icon(self.icon).classes('text-2xl mb-2').style(f'color: {self.color}')
            
            self.value_label = ui.label(str(self.value)).classes('metric-value').style(f'color: {self.color}')
            ui.label(self.label).classes('metric-label')
    
    def update(self, value: Any):
        """Update the metric value"""
        self.value = value
        if self.value_label:
            self.value_label.set_text(str(value))


class ChatMessage:
    """Agent chat message bubble"""
    
    AGENT_STYLES = {
        "scanner": (COLORS["scanner"], "🔍", "Scanner"),
        "analyzer": (COLORS["analyzer"], "🔬", "Analyzer"),
        "fixer": (COLORS["fixer"], "🔧", "Fixer"),
        "executor": (COLORS["executor"], "▶", "Executor"),
        "reporter": (COLORS["reporter"], "📊", "Reporter"),
        "llm": (COLORS["llm"], "🤖", "LLM"),
        "tool": (COLORS["tool"], "🛠", "Tool"),
        "user": (COLORS["primary"], "👤", "User"),
        "system": (COLORS["text_dim"], "⚙", "System"),
        "error": (COLORS["error"], "❌", "Error"),
    }
    
    def __init__(self, agent: str, content: str, turn: int = 0, has_tools: bool = False, tool_name: str = ""):
        color, icon, name = self.AGENT_STYLES.get(agent.lower(), (COLORS["text_secondary"], "💬", agent))
        is_tool = agent.lower() == "tool"
        
        with ui.element('div').classes('fade-in').style('margin: 8px 0;'):
            align = 'ml-auto' if is_tool else ''
            bubble_style = f'border-left: 3px solid {color};' if not is_tool else f'border-right: 3px solid {color};'
            
            with ui.element('div').classes(f'chat-bubble {"chat-bubble-right" if is_tool else "chat-bubble-left"} {align}').style(bubble_style):
                # Header
                with ui.element('div').classes('flex items-center gap-2 mb-2 pb-2').style('border-bottom: 1px solid rgba(255,255,255,0.1)'):
                    ui.label(f'{icon} {name}').classes('font-bold text-xs uppercase').style(f'color: {color}')
                    
                    if turn > 0:
                        ui.badge(f'T{turn}').props('outline').classes('text-xs')
                    
                    if has_tools:
                        ui.badge('TOOLS', color='orange').classes('text-xs')
                    
                    if tool_name:
                        ui.label(tool_name).classes('text-xs').style(f'color: {COLORS["info"]}')
                
                # Content
                display_content = content[:800] + "..." if len(content) > 800 else content
                ui.label(display_content).classes('text-sm mono').style(f'color: {COLORS["text_secondary"]}; white-space: pre-wrap;')


class FileTree:
    """Interactive file tree component"""
    
    def __init__(self, root_path: Path, on_select: Callable[[Path], None] = None):
        self.root_path = root_path
        self.on_select = on_select
        self.selected_path = None
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-3') as self.container:
            self.container.style(f'max-height: 400px; overflow-y: auto; background: {COLORS["bg_secondary"]};')
            self._render_tree(self.root_path, 0)
    
    def _render_tree(self, path: Path, indent: int):
        """Recursively render file tree"""
        ignore_dirs = {'__pycache__', '.git', 'venv', '.venv', 'node_modules', '.pytest_cache'}
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for item in items[:50]:  # Limit items
            if item.name.startswith('.') or item.name in ignore_dirs:
                continue
            
            indent_style = f'padding-left: {indent * 16}px'
            
            if item.is_dir():
                icon = "📁"
                icon_color = COLORS["primary"]
            else:
                ext = item.suffix.lower()
                if ext == '.py':
                    icon = "🐍"
                    icon_color = COLORS["success"]
                elif ext in ['.json', '.yaml', '.yml', '.toml']:
                    icon = "⚙"
                    icon_color = COLORS["warning"]
                elif ext == '.md':
                    icon = "📝"
                    icon_color = COLORS["info"]
                else:
                    icon = "📄"
                    icon_color = COLORS["text_secondary"]
            
            with ui.element('div').classes('file-tree-item').style(indent_style) as row:
                ui.label(icon).style(f'color: {icon_color}')
                ui.label(item.name).classes('flex-grow')
                
                if not item.is_dir():
                    row.on('click', lambda e, p=item: self._select_file(p))
            
            if item.is_dir() and indent < 2:
                self._render_tree(item, indent + 1)
    
    def _select_file(self, path: Path):
        """Handle file selection"""
        self.selected_path = path
        if self.on_select:
            self.on_select(path)
    
    def refresh(self, root_path: Path = None):
        """Refresh the tree"""
        if root_path:
            self.root_path = root_path
        if self.container:
            self.container.clear()
            with self.container:
                self._render_tree(self.root_path, 0)


class LogViewer:
    """Real-time log viewer with auto-scroll"""
    
    def __init__(self, max_logs: int = 100):
        self.max_logs = max_logs
        self.logs: List[Dict[str, Any]] = []
        self.container = None
        self._build()
    
    def _build(self):
        with ui.scroll_area().classes('glass-card w-full').style(
            f'height: 500px; width: 100%; background: {COLORS["bg_secondary"]};'
        ) as self.container:
            self._render_empty_state()
    
    def _render_empty_state(self):
        with ui.element('div').classes('flex items-center justify-center h-full'):
            ui.label('Waiting for logs...').style(f'color: {COLORS["text_dim"]}')
    
    def add_log(self, message: str, level: str = "info", agent: str = None, turn: int = 0):
        """Add a new log entry"""
        self.logs.append({
            "message": message,
            "level": level,
            "agent": agent,
            "turn": turn,
            "timestamp": datetime.now()
        })
        
        # Trim old logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        self._refresh()
    
    def add_chat(self, agent: str, content: str, turn: int = 0, has_tools: bool = False):
        """Add a chat message"""
        if self.container:
            with self.container:
                ChatMessage(agent, content, turn, has_tools)
            self.container.scroll_to(percent=1)
    
    def _refresh(self):
        """Refresh the log view"""
        if not self.container:
            return
        
        self.container.clear()
        with self.container:
            for log in self.logs[-50:]:
                level = log["level"]
                level_class = {
                    "success": "log-success",
                    "error": "log-error",
                    "warning": "log-warning",
                    "info": "log-info"
                }.get(level, "")
                
                with ui.element('div').classes(f'log-item {level_class} fade-in'):
                    ts = log["timestamp"].strftime("%H:%M:%S")
                    ui.label(f"[{ts}]").style(f'color: {COLORS["text_dim"]}')
                    ui.label(log["message"])
        
        # Auto-scroll to bottom
        self.container.scroll_to(percent=1)
    
    def clear(self):
        """Clear all logs"""
        self.logs = []
        if self.container:
            self.container.clear()
            with self.container:
                self._render_empty_state()


class WorkflowVisualizer:
    """Visualize LangGraph workflow state"""
    
    NODES = [
        ("scanner", "Scanner", COLORS["scanner"]),
        ("analyzer", "Analyzer", COLORS["analyzer"]),
        ("fixer", "Fixer", COLORS["fixer"]),
        ("executor", "Executor", COLORS["executor"]),
        ("reporter", "Reporter", COLORS["reporter"]),
    ]
    
    def __init__(self):
        self.current_node = None
        self.completed_nodes = set()
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4 w-full') as self.container:
            ui.label('Workflow Progress').classes('text-sm font-bold mb-4').style(f'color: {COLORS["text_dim"]}')
            
            with ui.element('div').classes('flex items-center justify-between gap-2'):
                for i, (node_id, name, color) in enumerate(self.NODES):
                    self._render_node(node_id, name, color)
                    
                    if i < len(self.NODES) - 1:
                        ui.element('div').style(f'width: 40px; height: 2px; background: {COLORS["border"]}')
    
    def _render_node(self, node_id: str, name: str, color: str):
        """Render a single workflow node"""
        is_active = node_id == self.current_node
        is_completed = node_id in self.completed_nodes
        
        classes = 'workflow-node'
        if is_active:
            classes += ' active pulse-glow'
        elif is_completed:
            classes += ' completed'
        
        border_color = color if is_active else (COLORS["success"] if is_completed else COLORS["border"])
        
        with ui.element('div').classes(classes).style(f'border-color: {border_color}'):
            if is_completed:
                ui.icon('check_circle').style(f'color: {COLORS["success"]}')
            elif is_active:
                ui.spinner(size='sm').style(f'color: {color}')
            else:
                ui.icon('radio_button_unchecked').style(f'color: {COLORS["text_dim"]}')
            
            ui.label(name).classes('text-xs font-medium mt-1').style(f'color: {color if is_active else COLORS["text_secondary"]}')
    
    def set_active(self, node_id: str):
        """Set the active node"""
        if self.current_node and self.current_node != node_id:
            self.completed_nodes.add(self.current_node)
        self.current_node = node_id
        self._refresh()
    
    def complete(self, node_id: str):
        """Mark a node as completed"""
        self.completed_nodes.add(node_id)
        if self.current_node == node_id:
            self.current_node = None
        self._refresh()
    
    def reset(self):
        """Reset workflow state"""
        self.current_node = None
        self.completed_nodes = set()
        self._refresh()
    
    def _refresh(self):
        """Refresh the visualization"""
        if self.container:
            self.container.clear()
            self.container.classes('glass-card p-4 w-full')
            with self.container:
                ui.label('Workflow Progress').classes('text-sm font-bold mb-4').style(f'color: {COLORS["text_dim"]}')
                
                with ui.element('div').classes('flex items-center justify-between gap-2'):
                    for i, (node_id, name, color) in enumerate(self.NODES):
                        self._render_node(node_id, name, color)
                        
                        if i < len(self.NODES) - 1:
                            line_color = COLORS["success"] if node_id in self.completed_nodes else COLORS["border"]
                            ui.element('div').style(f'width: 40px; height: 2px; background: {line_color}')


class CodeDiffViewer:
    """Side-by-side code diff viewer"""
    
    def __init__(self, original: str = "", modified: str = ""):
        self.original = original
        self.modified = modified
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4'):
            with ui.element('div').classes('flex gap-4'):
                # Original code
                with ui.element('div').classes('flex-1'):
                    ui.label('Original').classes('text-sm font-bold mb-2').style(f'color: {COLORS["error"]}')
                    with ui.element('div').classes('mono text-sm p-3 rounded').style(
                        f'background: {COLORS["code_bg"]}; max-height: 400px; overflow: auto;'
                    ):
                        ui.html(f'<pre style="margin: 0; color: {COLORS["text_secondary"]};">{self._escape_html(self.original)}</pre>')
                
                # Modified code
                with ui.element('div').classes('flex-1'):
                    ui.label('Modified').classes('text-sm font-bold mb-2').style(f'color: {COLORS["success"]}')
                    with ui.element('div').classes('mono text-sm p-3 rounded').style(
                        f'background: {COLORS["code_bg"]}; max-height: 400px; overflow: auto;'
                    ):
                        ui.html(f'<pre style="margin: 0; color: {COLORS["text_secondary"]};">{self._escape_html(self.modified)}</pre>')
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters"""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    def update(self, original: str, modified: str):
        """Update the diff view"""
        self.original = original
        self.modified = modified


class AIPromptInput:
    """AI prompt input with suggestions"""
    
    QUICK_PROMPTS = [
        "Fix all syntax errors in this file",
        "Optimize this code for performance",
        "Add type hints to all functions",
        "Write unit tests for this code",
        "Refactor to follow best practices",
        "Add error handling and logging",
    ]
    
    def __init__(self, on_submit: Callable[[str], None] = None):
        self.on_submit = on_submit
        self.input_field = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4'):
            ui.label('Ask AI Assistant').classes('text-sm font-bold mb-3').style(f'color: {COLORS["primary"]}')
            
            # Quick prompts
            with ui.element('div').classes('flex flex-wrap gap-2 mb-3'):
                for prompt in self.QUICK_PROMPTS[:4]:
                    short_prompt = prompt[:25] + "..." if len(prompt) > 25 else prompt
                    ui.button(short_prompt, on_click=lambda e, p=prompt: self._use_prompt(p)).props('outline dense').classes('text-xs')
            
            # Input area
            with ui.element('div').classes('flex gap-2'):
                self.input_field = ui.textarea(placeholder='Describe what you want to do with the code...').classes('flex-grow').props('outlined dense rows=2')
                ui.button(icon='send', on_click=self._submit).props('round color=primary')
    
    def _use_prompt(self, prompt: str):
        """Use a quick prompt"""
        if self.input_field:
            self.input_field.value = prompt
    
    def _submit(self):
        """Submit the prompt"""
        if self.input_field and self.input_field.value and self.on_submit:
            self.on_submit(self.input_field.value)
            self.input_field.value = ""

