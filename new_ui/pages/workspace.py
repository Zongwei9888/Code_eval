"""
Workspace Page - Full IDE-like experience
Combines code editor, terminal, and AI assistance in one view
"""
import sys
from pathlib import Path
from nicegui import ui
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from new_ui.theme import COLORS, GRADIENTS
from new_ui.components import (
    StatusIndicator, FileTree, LogViewer, 
    WorkflowVisualizer, ChatMessage
)
from new_ui.advanced_features import (
    TerminalEmulator, CodeCompletionPanel, 
    LintingPanel, CodeGenerationPanel
)

from config import DEFAULT_PROVIDER, MODEL_MAPPINGS


class WorkspacePage:
    """
    Full IDE-like workspace with multiple panes.
    """
    
    def __init__(self, bench_dir: Path):
        self.bench_dir = bench_dir
        self.current_project: Optional[str] = None
        self.current_file: Optional[str] = None
        self.original_code: str = ""
    
    def render(self):
        """Render the workspace page"""
        
        # Three-column layout
        with ui.splitter(value=20).classes('w-full h-screen') as main_splitter:
            main_splitter.style(f'background: {COLORS["bg_primary"]}')
            
            # Left sidebar - File explorer
            with ui.splitter_panel():
                self._render_explorer()
            
            # Main content area
            with ui.splitter_panel():
                with ui.splitter(value=70, horizontal=True).classes('h-full') as content_splitter:
                    
                    # Top - Code editor
                    with ui.splitter_panel():
                        self._render_editor()
                    
                    # Bottom - Terminal and output
                    with ui.splitter_panel():
                        self._render_bottom_panel()
            
            # Right sidebar - AI Assistant
            with ui.splitter(value=75).classes('h-full'):
                with ui.splitter_panel():
                    pass  # Empty, content is in main panel
                
                with ui.splitter_panel():
                    self._render_ai_panel()
    
    def _render_explorer(self):
        """Render file explorer sidebar"""
        with ui.element('div').classes('h-full p-2').style(f'background: {COLORS["bg_secondary"]}'):
            # Project selector
            ui.label('EXPLORER').classes('text-xs font-bold mb-2').style(f'color: {COLORS["text_dim"]}')
            
            projects = self._get_projects()
            self.project_select = ui.select(
                projects, 
                value=projects[0] if projects else None,
                on_change=self._on_project_change
            ).classes('w-full mb-4').props('dense outlined')
            
            # File tree
            if projects:
                project_path = self.bench_dir / projects[0]
                self.file_tree_container = ui.element('div')
                with self.file_tree_container:
                    FileTree(project_path, on_select=self._on_file_select)
    
    def _render_editor(self):
        """Render the code editor"""
        with ui.element('div').classes('h-full flex flex-col').style(f'background: {COLORS["bg_primary"]}'):
            # Editor tabs
            with ui.element('div').classes('flex items-center px-2').style(
                f'background: {COLORS["bg_secondary"]}; border-bottom: 1px solid {COLORS["border"]};'
            ):
                self.file_tab = ui.element('div').classes('px-4 py-2 text-sm').style(
                    f'background: {COLORS["bg_primary"]}; border-bottom: 2px solid {COLORS["primary"]};'
                )
                with self.file_tab:
                    ui.label('No file selected').style(f'color: {COLORS["text_secondary"]}')
            
            # Editor
            self.code_editor = ui.codemirror(
                value='# Select a file to edit',
                language='python',
                theme='oneDark'
            ).classes('flex-grow').style('font-size: 14px;')
            
            # Status bar
            with ui.element('div').classes('flex items-center justify-between px-4 py-1').style(
                f'background: {COLORS["bg_secondary"]}; border-top: 1px solid {COLORS["border"]};'
            ):
                ui.label('Python').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
                ui.label('UTF-8').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
                self.cursor_pos = ui.label('Ln 1, Col 1').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
    
    def _render_bottom_panel(self):
        """Render bottom panel with terminal and output"""
        with ui.element('div').classes('h-full').style(f'background: {COLORS["bg_secondary"]}'):
            with ui.tabs().classes('w-full') as tabs:
                term_tab = ui.tab('terminal', label='Terminal', icon='terminal')
                output_tab = ui.tab('output', label='Output', icon='output')
                problems_tab = ui.tab('problems', label='Problems', icon='error')
            
            with ui.tab_panels(tabs, value=term_tab).classes('w-full h-full'):
                with ui.tab_panel(term_tab):
                    TerminalEmulator()
                
                with ui.tab_panel(output_tab):
                    self.log_viewer = LogViewer()
                
                with ui.tab_panel(problems_tab):
                    LintingPanel()
    
    def _render_ai_panel(self):
        """Render AI assistant panel"""
        with ui.element('div').classes('h-full p-2').style(f'background: {COLORS["bg_secondary"]}'):
            ui.label('AI ASSISTANT').classes('text-xs font-bold mb-2').style(f'color: {COLORS["text_dim"]}')
            
            # Chat container
            with ui.element('div').classes('flex flex-col h-full'):
                # Messages
                self.chat_container = ui.scroll_area().classes('flex-grow').style(
                    f'background: {COLORS["bg_primary"]}; border-radius: 8px; padding: 8px;'
                )
                with self.chat_container:
                    ChatMessage("system", "Hi! I'm your AI coding assistant. Ask me anything!")
                
                # Input
                with ui.element('div').classes('mt-2'):
                    self.chat_input = ui.textarea(placeholder='Ask me anything...').props('outlined dense rows=2').classes('w-full')
                    ui.button('Send', on_click=self._send_chat).props('color=primary dense').classes('w-full mt-1')
    
    def _get_projects(self):
        """Get available projects"""
        if not self.bench_dir.exists():
            return []
        return sorted([
            d.name for d in self.bench_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__'
        ])
    
    def _on_project_change(self, e):
        """Handle project change"""
        self.current_project = e.value
        project_path = self.bench_dir / e.value
        
        self.file_tree_container.clear()
        with self.file_tree_container:
            FileTree(project_path, on_select=self._on_file_select)
    
    def _on_file_select(self, path: Path):
        """Handle file selection"""
        self.current_file = str(path)
        
        try:
            content = path.read_text(encoding='utf-8')
            self.original_code = content
            self.code_editor.value = content
            
            # Update tab
            self.file_tab.clear()
            with self.file_tab:
                ui.label(path.name).style(f'color: {COLORS["text_primary"]}')
        except Exception as e:
            ui.notify(f'Error loading file: {str(e)}', color='negative')
    
    def _send_chat(self):
        """Send chat message"""
        message = self.chat_input.value
        if not message:
            return
        
        with self.chat_container:
            ChatMessage("user", message)
            ChatMessage("llm", "I'm analyzing your request... (LLM integration coming soon!)", turn=1)
        
        self.chat_input.value = ""
        self.chat_container.scroll_to(percent=1)


def create_workspace_page(bench_dir: Path):
    """Create and render the workspace page"""
    workspace = WorkspacePage(bench_dir)
    workspace.render()
    return workspace

