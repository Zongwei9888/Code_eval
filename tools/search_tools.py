"""
Advanced Code Search and Navigation Tools

产品级代码搜索工具，参考 Cursor 和 Claude Code 的实现。

功能：
- grep_search: 正则表达式搜索（类似 ripgrep）
- find_definition: 查找符号定义
- find_references: 查找所有引用
- get_file_symbols: 获取文件符号/大纲
- semantic_search: 语义搜索（需要embedding）
"""

import os
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.tools import tool


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SearchMatch:
    """搜索匹配结果"""
    file: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int
    context_before: List[str]
    context_after: List[str]


@dataclass
class SymbolInfo:
    """符号信息"""
    name: str
    kind: str  # function, class, method, variable, import
    file: str
    line: int
    end_line: int
    signature: str
    docstring: Optional[str]


# ============================================================================
# GREP SEARCH
# ============================================================================

@tool
def grep_search(
    pattern: str,
    path: str = ".",
    file_pattern: str = None,
    case_sensitive: bool = True,
    context_lines: int = 2,
    max_results: int = 50,
    include_hidden: bool = False
) -> str:
    """
    使用正则表达式在代码中搜索。
    
    类似于 ripgrep (rg) 的功能，但直接集成在agent工具中。
    
    Args:
        pattern: 正则表达式模式
        path: 搜索路径（文件或目录）
        file_pattern: 文件名过滤（如 "*.py", "*.js"）
        case_sensitive: 是否区分大小写
        context_lines: 显示匹配行前后的上下文行数
        max_results: 最大返回结果数
        include_hidden: 是否包含隐藏文件
        
    Returns:
        格式化的搜索结果，包含文件路径、行号和匹配内容
        
    Example:
        # 搜索所有TODO注释
        grep_search(r"TODO:.*", path="./src")
        
        # 搜索函数定义
        grep_search(r"def\\s+process_\\w+", file_pattern="*.py")
        
        # 不区分大小写搜索
        grep_search("error", case_sensitive=False, context_lines=3)
    """
    try:
        root = Path(path)
        if not root.exists():
            return json.dumps({"error": f"路径不存在: {path}"})
        
        # 编译正则表达式
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return json.dumps({"error": f"无效的正则表达式: {e}"})
        
        # 忽略的目录
        ignore_dirs = {
            '__pycache__', '.git', 'node_modules', 'venv', '.venv',
            'build', 'dist', '.pytest_cache', '.mypy_cache', '.tox',
            '.idea', '.vscode', 'target', 'bin', 'obj'
        }
        
        matches = []
        files_searched = 0
        
        # 获取要搜索的文件
        if root.is_file():
            files = [root]
        else:
            if file_pattern:
                files = root.rglob(file_pattern)
            else:
                files = root.rglob("*")
        
        for file_path in files:
            # 跳过目录和忽略的路径
            if file_path.is_dir():
                continue
            if any(d in file_path.parts for d in ignore_dirs):
                continue
            if not include_hidden and file_path.name.startswith('.'):
                continue
            
            # 跳过二进制文件
            if _is_binary_file(file_path):
                continue
            
            files_searched += 1
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.splitlines()
                
                for i, line in enumerate(lines):
                    if regex.search(line):
                        # 获取上下文
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        context_before = lines[start:i]
                        context_after = lines[i+1:end]
                        
                        matches.append({
                            "file": str(file_path.relative_to(root) if path != "." else file_path),
                            "line": i + 1,
                            "content": line,
                            "context_before": context_before,
                            "context_after": context_after
                        })
                        
                        if len(matches) >= max_results:
                            break
                            
            except Exception:
                continue
            
            if len(matches) >= max_results:
                break
        
        # 格式化输出
        if not matches:
            return json.dumps({
                "matches": 0,
                "files_searched": files_searched,
                "message": f"未找到匹配 '{pattern}' 的内容"
            })
        
        output = {
            "pattern": pattern,
            "matches": len(matches),
            "files_searched": files_searched,
            "results": []
        }
        
        for m in matches:
            result = {
                "file": m["file"],
                "line": m["line"],
                "content": m["content"]
            }
            if context_lines > 0:
                result["context_before"] = m["context_before"]
                result["context_after"] = m["context_after"]
            output["results"].append(result)
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# SYMBOL SEARCH (AST-based for Python)
# ============================================================================

@tool
def find_definition(
    symbol_name: str,
    project_path: str = ".",
    language: str = "auto"
) -> str:
    """
    查找符号的定义位置。
    
    使用AST分析找到函数、类、方法、变量的定义。
    
    Args:
        symbol_name: 符号名称（函数名、类名、变量名）
        project_path: 项目路径
        language: 编程语言（auto自动检测，目前支持python）
        
    Returns:
        定义位置及其上下文
        
    Example:
        # 查找函数定义
        find_definition("process_data", "./src")
        
        # 查找类定义
        find_definition("UserService")
    """
    try:
        root = Path(project_path)
        if not root.exists():
            return json.dumps({"error": f"路径不存在: {project_path}"})
        
        definitions = []
        
        # 目前只实现Python
        for py_file in root.rglob("*.py"):
            if any(d in py_file.parts for d in ['__pycache__', '.git', 'venv', '.venv']):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    name = None
                    kind = None
                    signature = None
                    docstring = None
                    
                    if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
                        name = node.name
                        kind = "function"
                        signature = _get_function_signature(node)
                        docstring = ast.get_docstring(node)
                        
                    elif isinstance(node, ast.AsyncFunctionDef) and node.name == symbol_name:
                        name = node.name
                        kind = "async_function"
                        signature = _get_function_signature(node)
                        docstring = ast.get_docstring(node)
                        
                    elif isinstance(node, ast.ClassDef) and node.name == symbol_name:
                        name = node.name
                        kind = "class"
                        bases = [_get_name(b) for b in node.bases]
                        signature = f"class {name}({', '.join(bases)})"
                        docstring = ast.get_docstring(node)
                    
                    if name:
                        # 获取源代码上下文
                        lines = content.splitlines()
                        start_line = node.lineno
                        end_line = getattr(node, 'end_lineno', start_line + 5)
                        context = lines[start_line-1:min(end_line, start_line+10)]
                        
                        definitions.append({
                            "name": name,
                            "kind": kind,
                            "file": str(py_file.relative_to(root)),
                            "line": start_line,
                            "end_line": end_line,
                            "signature": signature,
                            "docstring": docstring[:200] if docstring else None,
                            "context": context[:10]  # 前10行
                        })
                        
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if not definitions:
            return json.dumps({
                "found": False,
                "message": f"未找到符号 '{symbol_name}' 的定义",
                "suggestion": "尝试使用 grep_search 进行文本搜索"
            })
        
        return json.dumps({
            "found": True,
            "symbol": symbol_name,
            "definitions": definitions
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def find_references(
    symbol_name: str,
    project_path: str = ".",
    include_definition: bool = False
) -> str:
    """
    查找符号的所有引用。
    
    找到代码库中所有使用该符号的位置。
    
    Args:
        symbol_name: 符号名称
        project_path: 项目路径
        include_definition: 是否包含定义位置
        
    Returns:
        所有引用位置的列表
        
    Example:
        # 找到所有调用process_data的地方
        find_references("process_data", "./src")
    """
    try:
        # 使用grep搜索所有引用
        # 构建匹配符号的正则（匹配完整单词）
        pattern = rf'\b{re.escape(symbol_name)}\b'
        
        result = grep_search(
            pattern=pattern,
            path=project_path,
            file_pattern="*.py",
            context_lines=1,
            max_results=100
        )
        
        result_data = json.loads(result)
        
        if "error" in result_data:
            return result
        
        # 如果不包含定义，需要过滤掉定义行
        if not include_definition and result_data.get("results"):
            filtered = []
            for r in result_data["results"]:
                content = r.get("content", "")
                # 简单判断是否是定义（def/class/=）
                is_definition = (
                    re.match(rf'^\s*(def|class)\s+{symbol_name}\b', content) or
                    re.match(rf'^\s*{symbol_name}\s*=', content)
                )
                if not is_definition:
                    filtered.append(r)
            result_data["results"] = filtered
            result_data["matches"] = len(filtered)
        
        return json.dumps(result_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_file_symbols(
    file_path: str
) -> str:
    """
    获取文件中的所有符号（大纲）。
    
    返回文件中定义的所有类、函数、方法、导入等。
    
    Args:
        file_path: 文件路径
        
    Returns:
        符号列表，包含类型、名称、行号、签名
        
    Example:
        get_file_symbols("main.py")
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"文件不存在: {file_path}"})
        
        content = path.read_text(encoding='utf-8')
        
        # 根据文件类型选择解析器
        if path.suffix == '.py':
            symbols = _parse_python_symbols(content)
        else:
            # 对于其他语言，使用简单的正则匹配
            symbols = _parse_generic_symbols(content, path.suffix)
        
        return json.dumps({
            "file": file_path,
            "symbols": symbols
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_files(
    pattern: str,
    path: str = ".",
    max_results: int = 50
) -> str:
    """
    按文件名搜索文件。
    
    Args:
        pattern: 文件名模式（支持通配符 * 和 ?）
        path: 搜索路径
        max_results: 最大返回结果数
        
    Returns:
        匹配的文件列表
        
    Example:
        # 搜索所有测试文件
        search_files("test_*.py")
        
        # 搜索配置文件
        search_files("*.config.js")
    """
    try:
        root = Path(path)
        if not root.exists():
            return json.dumps({"error": f"路径不存在: {path}"})
        
        ignore_dirs = {
            '__pycache__', '.git', 'node_modules', 'venv', '.venv',
            'build', 'dist'
        }
        
        matches = []
        for file_path in root.rglob(pattern):
            if file_path.is_dir():
                continue
            if any(d in file_path.parts for d in ignore_dirs):
                continue
            
            matches.append({
                "path": str(file_path.relative_to(root)),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })
            
            if len(matches) >= max_results:
                break
        
        return json.dumps({
            "pattern": pattern,
            "matches": len(matches),
            "files": matches
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_binary_file(file_path: Path) -> bool:
    """检查文件是否是二进制文件"""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except:
        return True


def _get_function_signature(node) -> str:
    """获取函数签名"""
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {_get_name(arg.annotation)}"
        args.append(arg_str)
    
    # 处理默认值
    defaults = node.args.defaults
    num_defaults = len(defaults)
    num_args = len(args)
    for i, default in enumerate(defaults):
        idx = num_args - num_defaults + i
        args[idx] += f"={_get_name(default)}"
    
    returns = ""
    if node.returns:
        returns = f" -> {_get_name(node.returns)}"
    
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){returns}"


def _get_name(node) -> str:
    """从AST节点获取名称字符串"""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Subscript):
        return f"{_get_name(node.value)}[{_get_name(node.slice)}]"
    elif isinstance(node, ast.BinOp):
        return f"{_get_name(node.left)} | {_get_name(node.right)}"
    else:
        return "..."


def _parse_python_symbols(content: str) -> List[Dict]:
    """解析Python文件的符号"""
    symbols = []
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return symbols
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append({
                    "name": alias.name,
                    "kind": "import",
                    "line": node.lineno
                })
                
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                symbols.append({
                    "name": f"{module}.{alias.name}" if module else alias.name,
                    "kind": "import",
                    "line": node.lineno
                })
                
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({
                "name": node.name,
                "kind": "function",
                "line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno),
                "signature": _get_function_signature(node)
            })
            
        elif isinstance(node, ast.ClassDef):
            # 类本身
            symbols.append({
                "name": node.name,
                "kind": "class",
                "line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)
            })
            
            # 类的方法
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({
                        "name": f"{node.name}.{item.name}",
                        "kind": "method",
                        "line": item.lineno,
                        "end_line": getattr(item, 'end_lineno', item.lineno),
                        "signature": _get_function_signature(item)
                    })
                    
        elif isinstance(node, ast.Assign):
            # 模块级变量
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append({
                        "name": target.id,
                        "kind": "variable",
                        "line": node.lineno
                    })
    
    return symbols


def _parse_generic_symbols(content: str, suffix: str) -> List[Dict]:
    """使用正则表达式解析其他语言的符号"""
    symbols = []
    lines = content.splitlines()
    
    patterns = {
        '.js': [
            (r'function\s+(\w+)', 'function'),
            (r'class\s+(\w+)', 'class'),
            (r'const\s+(\w+)\s*=', 'const'),
            (r'let\s+(\w+)\s*=', 'variable'),
        ],
        '.ts': [
            (r'function\s+(\w+)', 'function'),
            (r'class\s+(\w+)', 'class'),
            (r'interface\s+(\w+)', 'interface'),
            (r'type\s+(\w+)\s*=', 'type'),
            (r'const\s+(\w+)', 'const'),
        ],
        '.go': [
            (r'func\s+(\w+)', 'function'),
            (r'func\s+\([^)]+\)\s+(\w+)', 'method'),
            (r'type\s+(\w+)\s+struct', 'struct'),
            (r'type\s+(\w+)\s+interface', 'interface'),
        ],
        '.rs': [
            (r'fn\s+(\w+)', 'function'),
            (r'struct\s+(\w+)', 'struct'),
            (r'enum\s+(\w+)', 'enum'),
            (r'trait\s+(\w+)', 'trait'),
            (r'impl\s+(\w+)', 'impl'),
        ],
    }
    
    if suffix not in patterns:
        return symbols
    
    for i, line in enumerate(lines):
        for pattern, kind in patterns[suffix]:
            match = re.search(pattern, line)
            if match:
                symbols.append({
                    "name": match.group(1),
                    "kind": kind,
                    "line": i + 1
                })
    
    return symbols


# ============================================================================
# TOOL COLLECTION
# ============================================================================

SEARCH_TOOLS = [
    grep_search,
    find_definition,
    find_references,
    get_file_symbols,
    search_files
]

