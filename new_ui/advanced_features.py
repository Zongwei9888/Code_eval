"""
Advanced Features for Code Eval v4.0
Next-generation AI coding agent capabilities
"""
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from nicegui import ui

from .theme import COLORS, GRADIENTS


class CodeCompletionPanel:
    """
    AI-powered code completion suggestions panel.
    Shows context-aware suggestions as you type.
    """
    
    def __init__(self, on_accept: Callable[[str], None] = None):
        self.on_accept = on_accept
        self.suggestions: List[Dict[str, Any]] = []
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4') as self.container:
            with ui.element('div').classes('flex items-center gap-2 mb-3'):
                ui.icon('auto_fix_high').style(f'color: {COLORS["llm"]}')
                ui.label('AI Suggestions').classes('text-sm font-bold').style(f'color: {COLORS["llm"]}')
            
            self.suggestion_list = ui.element('div')
            with self.suggestion_list:
                ui.label('Start typing to see AI suggestions...').classes('text-sm').style(f'color: {COLORS["text_dim"]}')
    
    def update_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Update the suggestions list"""
        self.suggestions = suggestions
        self.suggestion_list.clear()
        
        with self.suggestion_list:
            if not suggestions:
                ui.label('No suggestions available').classes('text-sm').style(f'color: {COLORS["text_dim"]}')
                return
            
            for i, sugg in enumerate(suggestions[:5]):
                with ui.element('div').classes('p-3 rounded mb-2 hover-lift cursor-pointer').style(
                    f'background: {COLORS["bg_elevated"]}; border: 1px solid {COLORS["border"]};'
                ).on('click', lambda e, s=sugg: self._accept_suggestion(s)):
                    
                    # Suggestion type badge
                    sugg_type = sugg.get('type', 'code')
                    type_colors = {
                        'completion': COLORS["success"],
                        'fix': COLORS["error"],
                        'refactor': COLORS["warning"],
                        'optimize': COLORS["info"],
                    }
                    badge_color = type_colors.get(sugg_type, COLORS["primary"])
                    
                    with ui.element('div').classes('flex items-center gap-2 mb-2'):
                        ui.badge(sugg_type.upper(), color=badge_color).classes('text-xs')
                        ui.label(sugg.get('title', 'Suggestion')).classes('text-sm font-medium')
                    
                    # Code preview
                    code = sugg.get('code', '')[:100]
                    ui.label(code).classes('mono text-xs').style(
                        f'color: {COLORS["text_secondary"]}; background: {COLORS["code_bg"]}; padding: 8px; border-radius: 4px; display: block;'
                    )
    
    def _accept_suggestion(self, suggestion: Dict[str, Any]):
        """Accept a suggestion"""
        if self.on_accept:
            self.on_accept(suggestion.get('code', ''))
        ui.notify('Suggestion applied', color='positive')


class TerminalEmulator:
    """
    Embedded terminal emulator for running commands.
    Shows real-time output from code execution.
    """
    
    def __init__(self):
        self.output_lines: List[Dict[str, Any]] = []
        self.container = None
        self.is_running = False
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card').style(
            f'background: #0d1117; border: 1px solid {COLORS["border"]}; border-radius: 12px; overflow: hidden;'
        ) as self.container:
            
            # Terminal header
            with ui.element('div').classes('flex items-center gap-2 px-4 py-2').style(
                f'background: {COLORS["bg_elevated"]}; border-bottom: 1px solid {COLORS["border"]};'
            ):
                # Traffic light buttons
                for color in ['#ff5f56', '#ffbd2e', '#27c93f']:
                    ui.element('div').style(f'width: 12px; height: 12px; border-radius: 50%; background: {color};')
                
                ui.label('Terminal').classes('ml-4 text-sm mono').style(f'color: {COLORS["text_dim"]}')
                
                # Status indicator
                self.status_badge = ui.badge('READY', color='green').classes('ml-auto')
            
            # Terminal output
            self.output_container = ui.element('div').classes('p-4 mono text-sm').style(
                'height: 300px; overflow-y: auto;'
            )
            with self.output_container:
                self._render_prompt()
            
            # Input area
            with ui.element('div').classes('flex items-center px-4 py-2').style(
                f'border-top: 1px solid {COLORS["border"]};'
            ):
                ui.label('$').classes('mono mr-2').style(f'color: {COLORS["success"]}')
                self.command_input = ui.input(placeholder='Enter command...').props('dense borderless').classes('flex-grow mono')
                ui.button(icon='play_arrow', on_click=self._run_command).props('flat dense')
    
    def _render_prompt(self):
        """Render the shell prompt"""
        with ui.element('div').classes('flex items-center gap-2'):
            ui.label('$').style(f'color: {COLORS["success"]}')
            ui.label('_').classes('animate-pulse').style(f'color: {COLORS["text_primary"]}')
    
    def add_output(self, text: str, output_type: str = 'stdout'):
        """Add output to terminal"""
        color = {
            'stdout': COLORS["text_secondary"],
            'stderr': COLORS["error"],
            'system': COLORS["info"],
            'success': COLORS["success"],
        }.get(output_type, COLORS["text_secondary"])
        
        self.output_lines.append({'text': text, 'type': output_type, 'color': color})
        
        with self.output_container:
            ui.label(text).classes('mono').style(f'color: {color}; white-space: pre-wrap;')
        
        self.output_container.scroll_to(percent=1)
    
    def _run_command(self):
        """Run the entered command"""
        cmd = self.command_input.value
        if not cmd:
            return
        
        self.add_output(f'$ {cmd}', 'system')
        self.command_input.value = ''
        
        # Simulate command execution (in production, this would actually run the command)
        self.add_output('Command execution coming soon...', 'system')
    
    def clear(self):
        """Clear terminal output"""
        self.output_lines = []
        self.output_container.clear()
        with self.output_container:
            self._render_prompt()


class ProjectHealthDashboard:
    """
    Comprehensive project health dashboard.
    Shows code quality metrics, test coverage, and issues.
    """
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('grid grid-cols-2 lg:grid-cols-4 gap-4') as self.container:
            self._create_metric_card('health_score', 'Health Score', '-', COLORS["success"], 'favorite')
            self._create_metric_card('code_quality', 'Code Quality', '-', COLORS["info"], 'code')
            self._create_metric_card('test_coverage', 'Test Coverage', '-', COLORS["warning"], 'science')
            self._create_metric_card('issues', 'Open Issues', '-', COLORS["error"], 'bug_report')
    
    def _create_metric_card(self, key: str, label: str, value: str, color: str, icon: str):
        """Create a metric card"""
        with ui.element('div').classes('glass-card p-4 text-center hover-lift'):
            ui.icon(icon).classes('text-3xl mb-2').style(f'color: {color}')
            ui.label(value).classes('text-2xl font-bold cyber-heading').style(f'color: {color}')
            ui.label(label).classes('text-xs mt-1').style(f'color: {COLORS["text_dim"]}')
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update dashboard metrics"""
        self.metrics = metrics
        # In production, this would update the individual cards


class ConversationMemory:
    """
    Conversation memory component for multi-turn interactions.
    Stores and displays conversation history.
    """
    
    def __init__(self, max_messages: int = 50):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self.container = None
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.messages[-last_n:]
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def export(self) -> str:
        """Export conversation as JSON"""
        return json.dumps(self.messages, indent=2)


class CodeGenerationPanel:
    """
    Natural language to code generation panel.
    Uses LLM to generate code from descriptions.
    """
    
    def __init__(self, on_generate: Callable[[str], None] = None):
        self.on_generate = on_generate
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4') as self.container:
            with ui.element('div').classes('flex items-center gap-2 mb-4'):
                ui.icon('auto_awesome').style(f'color: {COLORS["warning"]}')
                ui.label('Code Generation').classes('text-lg font-bold').style(f'color: {COLORS["warning"]}')
            
            # Template buttons
            ui.label('Quick Templates').classes('text-xs mb-2').style(f'color: {COLORS["text_dim"]}')
            with ui.element('div').classes('flex flex-wrap gap-2 mb-4'):
                templates = [
                    ('API Endpoint', 'Create a FastAPI endpoint that...'),
                    ('Data Class', 'Create a dataclass for...'),
                    ('Unit Test', 'Write unit tests for...'),
                    ('Database Model', 'Create a SQLAlchemy model for...'),
                ]
                for name, prefix in templates:
                    ui.button(name, on_click=lambda e, p=prefix: self._use_template(p)).props('outline dense').classes('text-xs')
            
            # Description input
            ui.label('Describe what you want to create:').classes('text-sm mb-2').style(f'color: {COLORS["text_secondary"]}')
            self.description_input = ui.textarea(
                placeholder='e.g., Create a function that takes a list of numbers and returns the sum of all even numbers...'
            ).props('outlined rows=4').classes('w-full mb-4')
            
            # Language selection
            with ui.element('div').classes('flex items-center gap-4 mb-4'):
                ui.label('Language:').classes('text-sm').style(f'color: {COLORS["text_dim"]}')
                self.language_select = ui.select(
                    ['Python', 'JavaScript', 'TypeScript', 'Go', 'Rust'],
                    value='Python'
                ).props('dense outlined').classes('w-40')
            
            # Generate button
            ui.button('Generate Code', on_click=self._generate).props('color=warning').classes('w-full btn-primary')
            
            # Output area
            ui.separator().classes('my-4')
            ui.label('Generated Code').classes('text-sm font-bold mb-2').style(f'color: {COLORS["text_dim"]}')
            self.output_area = ui.element('div').classes('mono text-sm p-3 rounded').style(
                f'background: {COLORS["code_bg"]}; min-height: 150px;'
            )
            with self.output_area:
                ui.label('Generated code will appear here...').style(f'color: {COLORS["text_dim"]}')
    
    def _use_template(self, prefix: str):
        """Use a template"""
        self.description_input.value = prefix
    
    def _generate(self):
        """Generate code from description"""
        description = self.description_input.value
        if not description:
            ui.notify('Please enter a description', color='warning')
            return
        
        # In production, this would call the LLM
        self.output_area.clear()
        with self.output_area:
            ui.label('# Generated code would appear here\n# This feature uses the configured LLM\n').style(f'color: {COLORS["text_secondary"]}')
            ui.label(f'# Description: {description[:100]}...').style(f'color: {COLORS["text_dim"]}')
        
        if self.on_generate:
            self.on_generate(description)


class GitIntegrationPanel:
    """
    Git integration panel for version control operations.
    Shows git status, diffs, and allows commits.
    """
    
    def __init__(self, project_path: Path = None):
        self.project_path = project_path
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4') as self.container:
            with ui.element('div').classes('flex items-center justify-between mb-4'):
                with ui.element('div').classes('flex items-center gap-2'):
                    ui.icon('source').style(f'color: {COLORS["info"]}')
                    ui.label('Git Integration').classes('text-lg font-bold').style(f'color: {COLORS["info"]}')
                
                ui.badge('main', color='blue').props('outline')
            
            # Quick actions
            with ui.element('div').classes('flex gap-2 mb-4'):
                ui.button('Pull', icon='download').props('outline dense')
                ui.button('Push', icon='upload').props('outline dense')
                ui.button('Commit', icon='check').props('outline dense')
                ui.button('Stash', icon='archive').props('outline dense')
            
            # Status
            ui.label('Changes').classes('text-sm font-bold mb-2').style(f'color: {COLORS["text_dim"]}')
            with ui.element('div').style(f'max-height: 200px; overflow-y: auto;'):
                # Placeholder for git status
                for status, file, color in [
                    ('M', 'src/main.py', COLORS["warning"]),
                    ('A', 'src/utils.py', COLORS["success"]),
                    ('D', 'tests/old_test.py', COLORS["error"]),
                ]:
                    with ui.element('div').classes('flex items-center gap-2 py-1'):
                        ui.badge(status, color=color).classes('text-xs')
                        ui.label(file).classes('text-sm mono').style(f'color: {COLORS["text_secondary"]}')


class LintingPanel:
    """
    Real-time linting and code quality panel.
    Shows linter errors, warnings, and suggestions.
    """
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.container = None
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4') as self.container:
            with ui.element('div').classes('flex items-center justify-between mb-4'):
                with ui.element('div').classes('flex items-center gap-2'):
                    ui.icon('rule').style(f'color: {COLORS["warning"]}')
                    ui.label('Code Quality').classes('text-lg font-bold').style(f'color: {COLORS["warning"]}')
                
                self.issue_count = ui.badge('0 issues', color='green')
            
            # Filter tabs
            with ui.element('div').classes('flex gap-2 mb-4'):
                ui.button('All').props('flat dense')
                ui.button('Errors').props('flat dense')
                ui.button('Warnings').props('flat dense')
                ui.button('Info').props('flat dense')
            
            # Issues list
            self.issues_container = ui.element('div').style('max-height: 300px; overflow-y: auto;')
            with self.issues_container:
                ui.label('No issues found').classes('text-sm').style(f'color: {COLORS["success"]}')
    
    def update_issues(self, issues: List[Dict[str, Any]]):
        """Update the issues list"""
        self.issues = issues
        self.issue_count.set_text(f'{len(issues)} issues')
        self.issue_count.set_color('red' if issues else 'green')
        
        self.issues_container.clear()
        with self.issues_container:
            if not issues:
                ui.label('✓ No issues found').classes('text-sm').style(f'color: {COLORS["success"]}')
                return
            
            for issue in issues[:20]:
                severity = issue.get('severity', 'warning')
                color = {
                    'error': COLORS["error"],
                    'warning': COLORS["warning"],
                    'info': COLORS["info"],
                }.get(severity, COLORS["text_dim"])
                
                icon = {
                    'error': 'error',
                    'warning': 'warning',
                    'info': 'info',
                }.get(severity, 'help')
                
                with ui.element('div').classes('flex items-start gap-2 py-2').style(
                    f'border-bottom: 1px solid {COLORS["border"]};'
                ):
                    ui.icon(icon).style(f'color: {color}')
                    with ui.element('div').classes('flex-grow'):
                        ui.label(issue.get('message', 'Unknown issue')).classes('text-sm').style(f'color: {COLORS["text_secondary"]}')
                        with ui.element('div').classes('flex gap-2 mt-1'):
                            ui.label(f"Line {issue.get('line', '?')}").classes('text-xs mono').style(f'color: {COLORS["text_dim"]}')
                            ui.label(issue.get('rule', '')).classes('text-xs').style(f'color: {COLORS["text_dim"]}')


class KeyboardShortcutsPanel:
    """
    Keyboard shortcuts reference panel.
    """
    
    SHORTCUTS = [
        ('Ctrl+S', 'Save file'),
        ('Ctrl+Enter', 'Run analysis'),
        ('Ctrl+Shift+F', 'Format code'),
        ('Ctrl+/', 'Toggle comment'),
        ('Ctrl+Space', 'Trigger suggestions'),
        ('Ctrl+P', 'Quick file open'),
        ('Ctrl+G', 'Go to line'),
        ('F5', 'Start debugging'),
        ('Esc', 'Close panel'),
    ]
    
    def __init__(self):
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card p-4'):
            ui.label('Keyboard Shortcuts').classes('text-lg font-bold mb-4').style(f'color: {COLORS["primary"]}')
            
            for shortcut, description in self.SHORTCUTS:
                with ui.element('div').classes('flex items-center justify-between py-2').style(
                    f'border-bottom: 1px solid {COLORS["border"]};'
                ):
                    ui.label(description).classes('text-sm').style(f'color: {COLORS["text_secondary"]}')
                    ui.badge(shortcut).props('outline').classes('mono')

