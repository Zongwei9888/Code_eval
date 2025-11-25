"""
Local Repository Analysis Agents
Fast, non-LLM based utilities for project scanning and static analysis
"""
import os
import sys
import ast
import subprocess
import json
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FileAnalysis:
    """Analysis result for a single file"""
    path: str
    syntax_valid: bool = True
    syntax_errors: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    lines: int = 0
    complexity: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class ProjectInfo:
    """Project metadata and structure"""
    root_path: str
    name: str
    project_type: str = "unknown"
    has_requirements: bool = False
    has_setup_py: bool = False
    has_pyproject: bool = False
    has_tests: bool = False
    test_framework: str = ""
    entry_points: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    python_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result"""
    name: str
    passed: bool
    duration: float = 0.0
    error: str = ""
    output: str = ""


@dataclass
class AnalysisReport:
    """Complete analysis report"""
    project: ProjectInfo
    file_analyses: Dict[str, FileAnalysis] = field(default_factory=dict)
    test_results: List[TestResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    files_with_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = ""


# ============================================================================
# LOCAL AGENTS (Fast, No LLM Required)
# ============================================================================

class ProjectScannerAgent:
    """
    Agent for scanning and understanding project structure.
    Fast local analysis - no LLM calls required.
    """
    
    IGNORE_DIRS = {
        '__pycache__', '.git', '.svn', 'node_modules', 'venv', 'env',
        '.env', '.venv', 'build', 'dist', '.idea', '.vscode', 'eggs',
        '.tox', '.pytest_cache', '.mypy_cache', 'htmlcov', '.coverage'
    }
    
    def scan(self, project_path: str) -> ProjectInfo:
        """
        Scan project and extract metadata.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            ProjectInfo with discovered metadata
        """
        root = Path(project_path).resolve()
        
        info = ProjectInfo(
            root_path=str(root),
            name=root.name
        )
        
        # Check for Python project indicators
        info.has_requirements = (root / "requirements.txt").exists()
        info.has_setup_py = (root / "setup.py").exists()
        info.has_pyproject = (root / "pyproject.toml").exists()
        
        if info.has_requirements or info.has_setup_py or info.has_pyproject:
            info.project_type = "python"
        
        # Scan for config files
        config_patterns = ["*.toml", "*.cfg", "*.ini", "*.yaml", "*.yml", "*.json"]
        for pattern in config_patterns:
            for f in root.glob(pattern):
                if f.is_file():
                    info.config_files.append(f.name)
        
        # Scan for Python files
        for py_file in self._walk_files(root, ".py"):
            rel_path = str(py_file.relative_to(root))
            
            if self._is_test_file(py_file):
                info.test_files.append(rel_path)
                info.has_tests = True
            else:
                info.python_files.append(rel_path)
            
            # Check for entry points
            if py_file.name in ["main.py", "app.py", "__main__.py", "run.py"]:
                info.entry_points.append(rel_path)
        
        # Detect test framework
        if info.has_tests:
            info.test_framework = self._detect_test_framework(root)
        
        # Parse dependencies
        if info.has_requirements:
            info.dependencies = self._parse_requirements(root / "requirements.txt")
        
        return info
    
    def _walk_files(self, directory: Path, extension: str):
        """Walk directory yielding files with given extension"""
        try:
            for item in directory.iterdir():
                if item.name.startswith('.') or item.name in self.IGNORE_DIRS:
                    continue
                if item.is_dir():
                    yield from self._walk_files(item, extension)
                elif item.suffix == extension:
                    yield item
        except PermissionError:
            pass
    
    def _is_test_file(self, path: Path) -> bool:
        """Check if file is a test file"""
        name = path.name.lower()
        parts = [p.lower() for p in path.parts]
        
        return (
            name.startswith("test_") or
            name.endswith("_test.py") or
            name == "tests.py" or
            "tests" in parts or
            "test" in parts
        )
    
    def _detect_test_framework(self, root: Path) -> str:
        """Detect which test framework is used"""
        if (root / "pytest.ini").exists():
            return "pytest"
        
        if (root / "pyproject.toml").exists():
            content = (root / "pyproject.toml").read_text()
            if "[tool.pytest" in content:
                return "pytest"
        
        if (root / "requirements.txt").exists():
            content = (root / "requirements.txt").read_text().lower()
            if "pytest" in content:
                return "pytest"
        
        return "unittest"
    
    def _parse_requirements(self, req_file: Path) -> List[str]:
        """Parse requirements.txt"""
        deps = []
        try:
            for line in req_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
                    deps.append(pkg.strip())
        except Exception:
            pass
        return deps


class StaticAnalyzerAgent:
    """
    Agent for static code analysis.
    Fast local analysis using AST - no LLM calls required.
    """
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            FileAnalysis with results
        """
        path = Path(file_path)
        analysis = FileAnalysis(path=str(path))
        
        try:
            content = path.read_text(encoding='utf-8')
            analysis.lines = len(content.splitlines())
            
            try:
                tree = ast.parse(content)
                analysis.syntax_valid = True
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis.imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis.imports.append(node.module)
                    elif isinstance(node, ast.ClassDef):
                        analysis.classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        analysis.functions.append(node.name)
                
                # Calculate complexity
                analysis.complexity = self._calculate_complexity(tree)
                
            except SyntaxError as e:
                analysis.syntax_valid = False
                analysis.syntax_errors.append(f"Line {e.lineno}: {e.msg}")
                analysis.issues.append({
                    "type": "syntax_error",
                    "severity": "critical",
                    "line": e.lineno,
                    "message": e.msg
                })
                
        except Exception as e:
            analysis.issues.append({
                "type": "read_error",
                "severity": "critical",
                "message": str(e)
            })
        
        return analysis
    
    def analyze_files_parallel(
        self, 
        file_paths: List[str], 
        max_workers: int = 4
    ) -> Dict[str, FileAnalysis]:
        """
        Analyze multiple files in parallel.
        
        Args:
            file_paths: List of file paths to analyze
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file paths to their analyses
        """
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.analyze_file, path): path
                for path in file_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    results[path] = FileAnalysis(
                        path=path,
                        syntax_valid=False,
                        issues=[{"type": "analysis_error", "message": str(e)}]
                    )
        
        return results
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                ast.With, ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity


class EnvironmentAgent:
    """
    Agent for environment setup and dependency management.
    Handles virtual environments and package installation.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the environment agent.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        self.venv_path = self.project_path / ".venv"
    
    def check_environment(self) -> Dict[str, Any]:
        """Check current environment status"""
        return {
            "python_version": sys.version,
            "venv_exists": self.venv_path.exists(),
            "requirements_exists": (self.project_path / "requirements.txt").exists(),
            "installed_packages": self._get_installed_packages()
        }
    
    def setup_environment(self, use_venv: bool = False) -> Tuple[bool, str]:
        """
        Setup project environment.
        
        Args:
            use_venv: Whether to use a virtual environment
            
        Returns:
            Tuple of (success, message)
        """
        logs = []
        
        req_file = self.project_path / "requirements.txt"
        if not req_file.exists():
            return True, "No requirements.txt found, skipping dependency installation"
        
        try:
            logs.append("Installing dependencies from requirements.txt...")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_path)
            )
            
            if result.returncode == 0:
                logs.append("Dependencies installed successfully")
                return True, "\n".join(logs)
            else:
                logs.append(f"Installation failed: {result.stderr}")
                return False, "\n".join(logs)
                
        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _get_installed_packages(self) -> List[str]:
        """Get list of installed packages"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return [p["name"] for p in packages]
        except Exception:
            pass
        return []
    
    def check_missing_dependencies(self, required: List[str]) -> List[str]:
        """Check which dependencies are missing"""
        installed = set(p.lower() for p in self._get_installed_packages())
        missing = []
        
        for pkg in required:
            pkg_name = pkg.lower().split("[")[0]
            if pkg_name not in installed:
                missing.append(pkg)
        
        return missing


class TestRunnerAgent:
    """
    Agent for discovering and running tests.
    Supports pytest and unittest.
    """
    
    def __init__(self, project_path: str, framework: str = "pytest"):
        """
        Initialize the test runner agent.
        
        Args:
            project_path: Path to the project directory
            framework: Test framework to use ('pytest' or 'unittest')
        """
        self.project_path = Path(project_path)
        self.framework = framework
    
    def discover_tests(self) -> List[str]:
        """Discover all test files"""
        test_files = []
        
        for py_file in self.project_path.rglob("*.py"):
            name = py_file.name.lower()
            if name.startswith("test_") or name.endswith("_test.py"):
                test_files.append(str(py_file.relative_to(self.project_path)))
        
        return test_files
    
    def run_tests(
        self, 
        test_files: Optional[List[str]] = None, 
        timeout: int = 300
    ) -> Tuple[bool, List[TestResult], str]:
        """
        Run tests and collect results.
        
        Args:
            test_files: Specific test files to run (optional)
            timeout: Maximum test execution time
            
        Returns:
            Tuple of (success, results, output)
        """
        try:
            if self.framework == "pytest":
                return self._run_pytest(test_files, timeout)
            else:
                return self._run_unittest(test_files, timeout)
        except Exception as e:
            return False, [], f"Test execution error: {str(e)}"
    
    def _run_pytest(
        self, 
        test_files: Optional[List[str]], 
        timeout: int
    ) -> Tuple[bool, List[TestResult], str]:
        """Run tests using pytest"""
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"]
        
        if test_files:
            cmd.extend(test_files)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_path)
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            test_results = self._parse_pytest_output(output)
            
            return success, test_results, output
            
        except subprocess.TimeoutExpired:
            return False, [], "Tests timed out"
        except FileNotFoundError:
            return False, [], "pytest not installed"
    
    def _run_unittest(
        self, 
        test_files: Optional[List[str]], 
        timeout: int
    ) -> Tuple[bool, List[TestResult], str]:
        """Run tests using unittest"""
        cmd = [sys.executable, "-m", "unittest", "discover", "-v"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_path)
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            return success, [], output
            
        except subprocess.TimeoutExpired:
            return False, [], "Tests timed out"
    
    def _parse_pytest_output(self, output: str) -> List[TestResult]:
        """Parse pytest output to extract test results"""
        results = []
        
        for line in output.splitlines():
            if " PASSED" in line or " FAILED" in line or " ERROR" in line:
                parts = line.split()
                if parts:
                    name = parts[0]
                    passed = "PASSED" in line
                    error = line if not passed else ""
                    results.append(TestResult(
                        name=name,
                        passed=passed,
                        error=error
                    ))
        
        return results
    
    def run_single_file(
        self, 
        file_path: str, 
        timeout: int = 60
    ) -> Tuple[bool, str, str]:
        """
        Run a single Python file.
        
        Args:
            file_path: Path to the file
            timeout: Maximum execution time
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        full_path = self.project_path / file_path
        
        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_path)
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out"
        except Exception as e:
            return False, "", str(e)


# ============================================================================
# ORCHESTRATOR - Coordinates all local agents
# ============================================================================

class RepoAnalysisOrchestrator:
    """
    Orchestrates all local agents for repository analysis.
    Optimized for speed and efficiency with no LLM calls.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the orchestrator.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path)
        
        # Initialize local agents
        self.scanner = ProjectScannerAgent()
        self.static_analyzer = StaticAnalyzerAgent()
        
        # State
        self.project_info: Optional[ProjectInfo] = None
        self.report: Optional[AnalysisReport] = None
    
    def run_quick_check(self) -> Dict[str, Any]:
        """
        Run a quick check without LLM calls.
        
        Returns:
            Summary of issues found
        """
        # Scan
        self.project_info = self.scanner.scan(str(self.project_path))
        
        # Quick static analysis
        all_files = [str(self.project_path / f) for f in 
                    self.project_info.python_files + self.project_info.test_files]
        
        analyses = self.static_analyzer.analyze_files_parallel(all_files)
        
        # Summarize
        syntax_errors = []
        for path, analysis in analyses.items():
            if not analysis.syntax_valid:
                rel_path = str(Path(path).relative_to(self.project_path))
                syntax_errors.append({
                    "file": rel_path,
                    "errors": analysis.syntax_errors
                })
        
        return {
            "project_name": self.project_info.name,
            "total_files": len(all_files),
            "files_with_syntax_errors": len(syntax_errors),
            "syntax_errors": syntax_errors,
            "has_tests": self.project_info.has_tests,
            "test_framework": self.project_info.test_framework
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_scan(project_path: str) -> Dict[str, Any]:
    """
    Quickly scan a project without LLM calls.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Quick scan results
    """
    orchestrator = RepoAnalysisOrchestrator(project_path)
    return orchestrator.run_quick_check()


def create_orchestrator(project_path: str) -> RepoAnalysisOrchestrator:
    """
    Create a new orchestrator for project analysis.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        RepoAnalysisOrchestrator instance
    """
    return RepoAnalysisOrchestrator(project_path)
