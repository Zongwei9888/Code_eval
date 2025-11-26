"""
Dashboard Page - Overview and Quick Actions
Shows project stats, recent activity, and quick access features
"""
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from nicegui import ui

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from new_ui.theme import COLORS, GRADIENTS
from new_ui.components import MetricCard, StatusIndicator
from new_ui.advanced_features import ProjectHealthDashboard


class DashboardPage:
    """
    Main dashboard with project overview and quick actions.
    """
    
    def __init__(self, bench_dir: Path):
        self.bench_dir = bench_dir
    
    def render(self):
        """Render the dashboard"""
        with ui.element('div').classes('p-6').style(f'background: {COLORS["bg_primary"]}; min-height: 100vh;'):
            
            # Welcome header
            self._render_header()
            
            # Quick stats
            self._render_stats()
            
            # Main content grid
            with ui.element('div').classes('grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6'):
                # Recent projects
                self._render_recent_projects()
                
                # Quick actions
                self._render_quick_actions()
            
            # Activity feed
            self._render_activity_feed()
    
    def _render_header(self):
        """Render welcome header"""
        with ui.element('div').classes('glass-card p-6 mb-6').style(
            f'background: {GRADIENTS["hero"]}; border: 1px solid {COLORS["border"]};'
        ):
            with ui.element('div').classes('flex items-center justify-between'):
                with ui.element('div'):
                    ui.label('Welcome to Code Eval v4.0').classes('cyber-heading text-2xl font-bold').style(
                        f'color: {COLORS["text_primary"]}'
                    )
                    ui.label('AI-Powered Code Analysis & Improvement').classes('text-sm mt-1').style(
                        f'color: {COLORS["text_secondary"]}'
                    )
                
                with ui.element('div').classes('flex gap-2'):
                    ui.button('New Analysis', icon='add').props('color=primary').classes('btn-primary')
                    ui.button('Documentation', icon='book').props('flat')
    
    def _render_stats(self):
        """Render quick stats"""
        projects = self._get_projects()
        total_files = sum(len(list((self.bench_dir / p).rglob('*.py'))) for p in projects if (self.bench_dir / p).exists())
        
        with ui.element('div').classes('grid grid-cols-2 md:grid-cols-4 gap-4'):
            MetricCard(len(projects), 'PROJECTS', COLORS["primary"], 'folder')
            MetricCard(total_files, 'PYTHON FILES', COLORS["success"], 'description')
            MetricCard('0', 'ANALYSES TODAY', COLORS["warning"], 'analytics')
            MetricCard('Active', 'SYSTEM STATUS', COLORS["info"], 'power_settings_new')
    
    def _render_recent_projects(self):
        """Render recent projects list"""
        with ui.element('div').classes('glass-card p-4'):
            with ui.element('div').classes('flex items-center justify-between mb-4'):
                ui.label('Recent Projects').classes('text-lg font-bold').style(f'color: {COLORS["text_primary"]}')
                ui.button('View All', icon='arrow_forward').props('flat dense')
            
            projects = self._get_projects()[:5]
            
            for project in projects:
                project_path = self.bench_dir / project
                file_count = len(list(project_path.rglob('*.py'))) if project_path.exists() else 0
                
                with ui.element('div').classes('flex items-center justify-between py-3 hover-lift cursor-pointer').style(
                    f'border-bottom: 1px solid {COLORS["border"]};'
                ):
                    with ui.element('div').classes('flex items-center gap-3'):
                        ui.icon('folder').style(f'color: {COLORS["primary"]}')
                        with ui.element('div'):
                            ui.label(project).classes('font-medium').style(f'color: {COLORS["text_primary"]}')
                            ui.label(f'{file_count} Python files').classes('text-xs').style(f'color: {COLORS["text_dim"]}')
                    
                    ui.button(icon='play_arrow').props('flat round dense')
    
    def _render_quick_actions(self):
        """Render quick action buttons"""
        with ui.element('div').classes('glass-card p-4'):
            ui.label('Quick Actions').classes('text-lg font-bold mb-4').style(f'color: {COLORS["text_primary"]}')
            
            actions = [
                ('Single File Analysis', 'description', COLORS["scanner"], '/'),
                ('Quick Scan', 'search', COLORS["analyzer"], '/'),
                ('Multi-Agent Workflow', 'psychology', COLORS["fixer"], '/'),
                ('AI Chat', 'chat', COLORS["llm"], '/'),
                ('Code Generation', 'auto_awesome', COLORS["warning"], '/'),
                ('Settings', 'settings', COLORS["text_dim"], '/'),
            ]
            
            with ui.element('div').classes('grid grid-cols-2 gap-3'):
                for name, icon, color, link in actions:
                    with ui.element('div').classes('glass-card p-4 text-center hover-lift cursor-pointer').on('click', lambda: ui.navigate.to(link)):
                        ui.icon(icon).classes('text-3xl mb-2').style(f'color: {color}')
                        ui.label(name).classes('text-sm').style(f'color: {COLORS["text_secondary"]}')
    
    def _render_activity_feed(self):
        """Render recent activity feed"""
        with ui.element('div').classes('glass-card p-4 mt-6'):
            with ui.element('div').classes('flex items-center justify-between mb-4'):
                ui.label('Recent Activity').classes('text-lg font-bold').style(f'color: {COLORS["text_primary"]}')
                ui.button('Clear All', icon='clear_all').props('flat dense')
            
            # Placeholder activities
            activities = [
                ('Analysis completed', 'check_circle', COLORS["success"], '2 minutes ago'),
                ('Code fixed successfully', 'build', COLORS["fixer"], '15 minutes ago'),
                ('New project added', 'add', COLORS["primary"], '1 hour ago'),
                ('Workflow started', 'play_arrow', COLORS["info"], '2 hours ago'),
            ]
            
            for text, icon, color, time in activities:
                with ui.element('div').classes('flex items-center gap-3 py-2').style(
                    f'border-bottom: 1px solid {COLORS["border"]};'
                ):
                    ui.icon(icon).style(f'color: {color}')
                    ui.label(text).classes('flex-grow').style(f'color: {COLORS["text_secondary"]}')
                    ui.label(time).classes('text-xs').style(f'color: {COLORS["text_dim"]}')
    
    def _get_projects(self) -> List[str]:
        """Get available projects"""
        if not self.bench_dir.exists():
            return []
        return sorted([
            d.name for d in self.bench_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__'
        ])


def create_dashboard_page(bench_dir: Path):
    """Create and render the dashboard page"""
    dashboard = DashboardPage(bench_dir)
    dashboard.render()
    return dashboard

