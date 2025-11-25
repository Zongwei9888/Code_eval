"""
Project-level code analyzer
Supports scanning entire repositories and analyzing multiple files
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class FileInfo:
    """Information about a single file"""
    path: str
    relative_path: str
    size: int
    lines: int
    language: str = "python"
    issues: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, analyzing, success, error


@dataclass
class ProjectInfo:
    """Information about a project"""
    root_path: str
    name: str
    total_files: int = 0
    total_lines: int = 0
    files: List[FileInfo] = field(default_factory=list)
    analyzed_files: int = 0
    fixed_files: int = 0


class ProjectAnalyzer:
    """
    Analyzes entire project directories
    Supports repo-level code analysis and modification
    """
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
    }
    
    IGNORE_DIRS = {
        '__pycache__', '.git', '.svn', 'node_modules', 'venv', 
        'env', '.env', '.venv', 'build', 'dist', '.idea', 
        '.vscode', 'eggs', '*.egg-info', '.tox', '.pytest_cache',
        '.mypy_cache', '.coverage', 'htmlcov'
    }
    
    IGNORE_FILES = {
        '.gitignore', '.dockerignore', 'LICENSE', 'README.md',
        'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml'
    }
    
    def __init__(self, root_path: str):
        """
        Initialize project analyzer
        
        Args:
            root_path: Root directory of the project
        """
        self.root_path = Path(root_path).resolve()
        self.project_info: Optional[ProjectInfo] = None
        
    def scan_project(self, extensions: Optional[List[str]] = None) -> ProjectInfo:
        """
        Scan project directory and collect file information
        
        Args:
            extensions: List of file extensions to include (e.g., ['.py', '.js'])
                       If None, scans all supported extensions
                       
        Returns:
            ProjectInfo with all discovered files
        """
        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS.keys())
        
        files = []
        total_lines = 0
        
        for file_path in self._walk_directory(self.root_path):
            if file_path.suffix.lower() in extensions:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                    file_info = FileInfo(
                        path=str(file_path),
                        relative_path=str(file_path.relative_to(self.root_path)),
                        size=file_path.stat().st_size,
                        lines=lines,
                        language=self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown')
                    )
                    files.append(file_info)
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        self.project_info = ProjectInfo(
            root_path=str(self.root_path),
            name=self.root_path.name,
            total_files=len(files),
            total_lines=total_lines,
            files=files
        )
        
        return self.project_info
    
    def _walk_directory(self, directory: Path):
        """
        Walk directory, respecting ignore patterns
        
        Yields:
            Path objects for each file
        """
        try:
            for item in directory.iterdir():
                # Skip hidden files/directories
                if item.name.startswith('.'):
                    continue
                    
                # Skip ignored directories
                if item.is_dir():
                    if item.name in self.IGNORE_DIRS:
                        continue
                    yield from self._walk_directory(item)
                else:
                    # Skip ignored files
                    if item.name in self.IGNORE_FILES:
                        continue
                    yield item
        except PermissionError:
            pass
    
    def get_file_content(self, file_path: str) -> str:
        """
        Get content of a specific file
        
        Args:
            file_path: Path to the file (relative or absolute)
            
        Returns:
            File content as string
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root_path / path
            
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def get_file_tree(self) -> Dict[str, Any]:
        """
        Get project file tree structure
        
        Returns:
            Nested dictionary representing file tree
        """
        if not self.project_info:
            self.scan_project()
            
        tree = {"name": self.project_info.name, "type": "directory", "children": []}
        
        for file_info in self.project_info.files:
            parts = Path(file_info.relative_path).parts
            current = tree["children"]
            
            # Build path
            for i, part in enumerate(parts[:-1]):
                # Find or create directory
                found = None
                for child in current:
                    if child["name"] == part and child["type"] == "directory":
                        found = child
                        break
                
                if not found:
                    found = {"name": part, "type": "directory", "children": []}
                    current.append(found)
                
                current = found["children"]
            
            # Add file
            current.append({
                "name": parts[-1],
                "type": "file",
                "path": file_info.relative_path,
                "lines": file_info.lines,
                "status": file_info.status
            })
        
        return tree
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get project summary statistics
        
        Returns:
            Dictionary with project statistics
        """
        if not self.project_info:
            self.scan_project()
            
        # Count by language
        by_language = {}
        for f in self.project_info.files:
            lang = f.language
            if lang not in by_language:
                by_language[lang] = {"files": 0, "lines": 0}
            by_language[lang]["files"] += 1
            by_language[lang]["lines"] += f.lines
        
        return {
            "name": self.project_info.name,
            "root_path": self.project_info.root_path,
            "total_files": self.project_info.total_files,
            "total_lines": self.project_info.total_lines,
            "by_language": by_language,
            "analyzed": self.project_info.analyzed_files,
            "fixed": self.project_info.fixed_files
        }
    
    def update_file_status(self, relative_path: str, status: str, issues: List[str] = None):
        """
        Update status of a specific file
        
        Args:
            relative_path: Relative path to the file
            status: New status (pending, analyzing, success, error)
            issues: List of issues found (optional)
        """
        if not self.project_info:
            return
            
        for file_info in self.project_info.files:
            if file_info.relative_path == relative_path:
                file_info.status = status
                if issues is not None:
                    file_info.issues = issues
                    
                # Update counters
                if status == "success":
                    self.project_info.analyzed_files += 1
                    if issues and len(issues) > 0:
                        self.project_info.fixed_files += 1
                break


def scan_code_bench(bench_path: str = "code_bench") -> List[Dict[str, Any]]:
    """
    Scan code_bench directory for projects
    
    Args:
        bench_path: Path to code_bench directory
        
    Returns:
        List of project information dictionaries
    """
    bench = Path(bench_path)
    if not bench.exists():
        bench.mkdir(parents=True, exist_ok=True)
        return []
    
    projects = []
    
    for item in bench.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            analyzer = ProjectAnalyzer(str(item))
            info = analyzer.scan_project(['.py'])  # Focus on Python for now
            projects.append(analyzer.get_summary())
    
    return projects

