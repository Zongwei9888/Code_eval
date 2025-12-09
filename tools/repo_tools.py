"""
Repository Analysis Tools
Consolidated tools for multi-agent repository analysis workflow
"""
import os
import sys
import ast
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool


# ============================================================================
# PROJECT SCANNING TOOLS
# ============================================================================

@tool
def scan_project(project_path: str, language: str = "all") -> str:
    """
    Scan a project directory to find source code files.
    
    Supports multiple languages: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, etc.
    
    Args:
        project_path: Path to the project directory
        language: Language to scan for ("python", "javascript", "typescript", "go", "rust", "java", "cpp", "all")
                 Default is "all" to scan all supported languages
        
    Returns:
        JSON string with source_files, test_files, entry_points, and file counts by language
        
    Example:
        # Scan Python files only
        scan_project("/path/to/project", language="python")
        
        # Scan all languages
        scan_project("/path/to/project", language="all")
    """
    root = Path(project_path)
    if not root.exists():
        return json.dumps({"error": f"Path does not exist: {project_path}"})
    
    ignore_dirs = {
        '__pycache__', '.git', 'venv', '.venv', 'node_modules',
        'build', 'dist', '.pytest_cache', '.mypy_cache', '.tox',
        'target', 'vendor', 'bin', 'obj', '.next', 'out'
    }
    
    # 定义语言文件扩展名和入口点模式
    language_patterns = {
        "python": {
            "extensions": [".py"],
            "entry_points": ["main.py", "app.py", "run.py", "__main__.py", "manage.py", "cli.py"],
            "test_patterns": ["test_*.py", "*_test.py", "tests/"]
        },
        "javascript": {
            "extensions": [".js", ".jsx", ".mjs"],
            "entry_points": ["index.js", "main.js", "app.js", "server.js"],
            "test_patterns": ["*.test.js", "*.spec.js", "__tests__/"]
        },
        "typescript": {
            "extensions": [".ts", ".tsx"],
            "entry_points": ["index.ts", "main.ts", "app.ts", "server.ts"],
            "test_patterns": ["*.test.ts", "*.spec.ts", "__tests__/"]
        },
        "go": {
            "extensions": [".go"],
            "entry_points": ["main.go"],
            "test_patterns": ["*_test.go"]
        },
        "rust": {
            "extensions": [".rs"],
            "entry_points": ["main.rs", "lib.rs"],
            "test_patterns": ["tests/"]
        },
        "java": {
            "extensions": [".java"],
            "entry_points": ["Main.java", "Application.java"],
            "test_patterns": ["*Test.java", "test/"]
        },
        "cpp": {
            "extensions": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"],
            "entry_points": ["main.cpp", "main.c", "main.cc"],
            "test_patterns": ["*_test.cpp", "test/"]
        }
    }
    
    # 确定要扫描的语言
    if language.lower() == "all":
        languages_to_scan = language_patterns.keys()
    elif language.lower() in language_patterns:
        languages_to_scan = [language.lower()]
    else:
        return json.dumps({"error": f"Unsupported language: {language}. Supported: {list(language_patterns.keys())}"})
    
    # 收集文件
    source_files = []
    test_files = []
    entry_points = []
    files_by_language = {}
    
    for lang in languages_to_scan:
        lang_config = language_patterns[lang]
        lang_files = []
        
        for ext in lang_config["extensions"]:
            for file_path in root.rglob(f"*{ext}"):
                # 跳过忽略目录
                if any(d in file_path.parts for d in ignore_dirs):
                    continue
                
                rel_path = str(file_path.relative_to(root))
                lang_files.append(rel_path)
                
                # 检查是否是测试文件
                is_test = any(
                    pattern.rstrip('/') in rel_path 
                    for pattern in lang_config["test_patterns"]
                )
                
                if is_test:
                    test_files.append({"file": rel_path, "language": lang})
                else:
                    source_files.append({"file": rel_path, "language": lang})
                
                # 检查是否是入口点
                if file_path.name in lang_config["entry_points"]:
                    entry_points.append({"file": rel_path, "language": lang})
        
        files_by_language[lang] = len(lang_files)
    
    # 对于Python，保持向后兼容的格式
    python_files = [f["file"] for f in source_files if f["language"] == "python"]
    python_test_files = [f["file"] for f in test_files if f["language"] == "python"]
    
    return json.dumps({
        "project_path": str(root),
        "language": language,
        # 向后兼容字段（Python）
        "python_files": sorted(python_files),
        "test_files": sorted(python_test_files),
        # 新增多语言字段
        "source_files": sorted(source_files, key=lambda x: x["file"]),
        "all_test_files": sorted(test_files, key=lambda x: x["file"]),
        "entry_points": sorted(entry_points, key=lambda x: x["file"]),
        "files_by_language": files_by_language,
        "total_files": len(source_files) + len(test_files),
        "total_source_files": len(source_files),
        "total_test_files": len(test_files)
    }, indent=2)


# ============================================================================
# FILE OPERATION TOOLS
# ============================================================================

@tool
def read_file_content(file_path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content with metadata or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        content = path.read_text(encoding='utf-8')
        return f"=== FILE: {file_path} ===\n=== LINES: {len(content.splitlines())} ===\n\n{content}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def write_file_content(file_path: str, content: str) -> str:
    """
    Write content to a file. Use this to fix code.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"SUCCESS: Written {len(content)} characters to {file_path}"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# CODE ANALYSIS TOOLS
# ============================================================================

@tool
def check_python_syntax(file_path: str) -> str:
    """
    Check Python syntax of a file using AST parser.
    
    Args:
        file_path: Path to the Python file to check
        
    Returns:
        JSON with syntax check results including classes, functions, and imports
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"valid": False, "error": "File not found", "file": file_path})
        
        content = path.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
            
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            
            return json.dumps({
                "valid": True,
                "file": file_path,
                "lines": len(content.splitlines()),
                "classes": classes,
                "functions": functions,
                "imports": imports
            }, indent=2)
            
        except SyntaxError as e:
            return json.dumps({
                "valid": False,
                "file": file_path,
                "error": f"Line {e.lineno}: {e.msg}",
                "line_number": e.lineno,
                "offset": e.offset
            }, indent=2)
            
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e), "file": file_path})


# ============================================================================
# CODE EXECUTION TOOLS
# ============================================================================

@tool
def execute_python_file(file_path: str, timeout: int = 30) -> str:
    """
    Execute a Python file and capture stdout/stderr.
    
    Args:
        file_path: Path to the Python file to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        JSON with success status, output, and any errors
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": "File not found"})
        
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(path.parent)
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[:3000] if result.stdout else "",
            "stderr": result.stderr[:3000] if result.stderr else "",
            "file": file_path
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Timeout", "file": file_path})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "file": file_path})


# ============================================================================
# TEST EXECUTION TOOLS
# ============================================================================

@tool
def run_pytest(project_path: str, timeout: int = 120) -> str:
    """
    Run pytest on a project and return results.
    
    Args:
        project_path: Path to the project directory
        timeout: Maximum test execution time in seconds
        
    Returns:
        JSON with test results including pass/fail counts
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_path
        )
        
        output = result.stdout + result.stderr
        passed = output.count(" PASSED") + output.count(" passed")
        failed = output.count(" FAILED") + output.count(" failed") + output.count(" ERROR")
        
        return json.dumps({
            "success": result.returncode == 0,
            "passed": passed,
            "failed": failed,
            "output": output[:4000]
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Tests timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def run_unittest(project_path: str, timeout: int = 120) -> str:
    """
    Run unittest discover on a project.
    
    Args:
        project_path: Path to the project directory
        timeout: Maximum test execution time in seconds
        
    Returns:
        JSON with test results
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_path
        )
        
        output = result.stdout + result.stderr
        
        return json.dumps({
            "success": result.returncode == 0,
            "output": output[:4000]
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Tests timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# DEPENDENCY TOOLS
# ============================================================================

@tool
def install_dependencies(project_path: str, timeout: int = 300) -> str:
    """
    Install project dependencies from requirements.txt.
    
    Args:
        project_path: Path to the project directory
        timeout: Maximum installation time in seconds
        
    Returns:
        JSON with installation result
    """
    req_file = Path(project_path) / "requirements.txt"
    
    if not req_file.exists():
        return json.dumps({"success": True, "message": "No requirements.txt found"})
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_path
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "message": "Dependencies installed" if result.returncode == 0 else result.stderr[:500]
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Installation timed out"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# TOOL COLLECTIONS
# ============================================================================

# All repository analysis tools
ALL_REPO_TOOLS = [
    scan_project,
    read_file_content,
    write_file_content,
    check_python_syntax,
    execute_python_file,
    run_pytest,
    run_unittest,
    install_dependencies
]

# Scanner tools subset
SCANNER_TOOLS = [scan_project]

# Analyzer tools subset
ANALYZER_TOOLS = [check_python_syntax, read_file_content]

# Fixer tools subset
FIXER_TOOLS = [read_file_content, write_file_content]

# Executor tools subset
EXECUTOR_TOOLS = [execute_python_file, run_pytest, run_unittest]


def get_repo_tool_by_name(tool_name: str):
    """
    Get a tool by its name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool instance or None if not found
    """
    for tool in ALL_REPO_TOOLS:
        if tool.name == tool_name:
            return tool
    return None

