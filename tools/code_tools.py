"""
Tools for multi-agent coding assistant
Includes file operations, code execution, and analysis tools
"""
import os
import subprocess
import sys
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import traceback
from langchain.tools import tool
from pydantic import BaseModel, Field


class CodeExecutionResult(BaseModel):
    """Result of code execution"""
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int = 0


class FileOperation(BaseModel):
    """File operation parameters"""
    file_path: str = Field(description="Path to the file")
    content: Optional[str] = Field(None, description="Content to write to file")


class MCPRequest(BaseModel):
    """MCP tool request parameters"""
    tool_name: str = Field(description="Name of the MCP tool to call")
    parameters: Dict[str, Any] = Field(description="Parameters for the MCP tool")


@tool
def read_file_tool(file_path: str) -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Content of the file or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"File content ({len(content)} characters):\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}\n{traceback.format_exc()}"


@tool
def write_file_tool(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}\n{traceback.format_exc()}"


@tool
def list_files_tool(directory: str = ".") -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory path to list files from
        
    Returns:
        List of files or error message
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory {directory} does not exist"
        
        files = []
        for item in path.rglob("*"):
            if item.is_file() and not any(part.startswith('.') for part in item.parts):
                files.append(str(item.relative_to(path)))
        
        return f"Files in {directory}:\n" + "\n".join(f"  - {f}" for f in sorted(files))
    except Exception as e:
        return f"Error listing files: {str(e)}\n{traceback.format_exc()}"


@tool
def execute_python_code(code: str, timeout: int = 30) -> str:
    """
    Execute Python code and capture output/errors.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result with output and errors
    """
    try:
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout
            error = result.stderr
            exit_code = result.returncode
            
            # Format result
            if exit_code == 0:
                return f"SUCCESS:\nOutput:\n{output}\n\nStderr (if any):\n{error if error else 'None'}"
            else:
                return f"FAILED (exit code {exit_code}):\nOutput:\n{output}\n\nError:\n{error}"
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return f"ERROR: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"


@tool
def execute_file(file_path: str, timeout: int = 30) -> str:
    """
    Execute a Python file and capture output/errors.
    
    Args:
        file_path: Path to Python file to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Execution result with output and errors
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=path.parent
        )
        
        output = result.stdout
        error = result.stderr
        exit_code = result.returncode
        
        if exit_code == 0:
            return f"SUCCESS:\nOutput:\n{output}\n\nStderr (if any):\n{error if error else 'None'}"
        else:
            return f"FAILED (exit code {exit_code}):\nOutput:\n{output}\n\nError:\n{error}"
            
    except subprocess.TimeoutExpired:
        return f"ERROR: Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"


@tool
def analyze_error(error_message: str) -> str:
    """
    Analyze error message and suggest fixes.
    
    Args:
        error_message: Error message from code execution
        
    Returns:
        Analysis and suggestions
    """
    analysis = ["Error Analysis:"]
    suggestions = []
    
    # Common error patterns
    if "ModuleNotFoundError" in error_message or "ImportError" in error_message:
        analysis.append("- Missing module/package")
        suggestions.append("Install required package using pip")
        suggestions.append("Check import statement spelling")
        
    elif "SyntaxError" in error_message:
        analysis.append("- Syntax error in code")
        suggestions.append("Check for missing colons, parentheses, or brackets")
        suggestions.append("Verify indentation")
        
    elif "NameError" in error_message:
        analysis.append("- Variable or function not defined")
        suggestions.append("Check variable name spelling")
        suggestions.append("Ensure variable is defined before use")
        
    elif "TypeError" in error_message:
        analysis.append("- Type mismatch")
        suggestions.append("Check function arguments types")
        suggestions.append("Verify operation compatibility with data types")
        
    elif "AttributeError" in error_message:
        analysis.append("- Attribute does not exist")
        suggestions.append("Check object type")
        suggestions.append("Verify attribute name spelling")
        
    elif "IndentationError" in error_message:
        analysis.append("- Incorrect indentation")
        suggestions.append("Use consistent indentation (spaces or tabs)")
        suggestions.append("Check block structure")
        
    else:
        analysis.append("- General error")
        suggestions.append("Review error traceback carefully")
    
    result = "\n".join(analysis) + "\n\nSuggestions:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggestions))
    return result


@tool
def search_code(directory: str, search_term: str) -> str:
    """
    Search for a term in code files.
    
    Args:
        directory: Directory to search in
        search_term: Term to search for
        
    Returns:
        Files containing the search term
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory {directory} does not exist"
        
        matches = []
        for file_path in path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if search_term in content:
                        lines = content.split('\n')
                        matching_lines = [
                            (i+1, line) for i, line in enumerate(lines) 
                            if search_term in line
                        ]
                        matches.append({
                            'file': str(file_path.relative_to(path)),
                            'occurrences': matching_lines
                        })
            except Exception:
                continue
        
        if not matches:
            return f"No matches found for '{search_term}' in {directory}"
        
        result = [f"Found '{search_term}' in {len(matches)} file(s):"]
        for match in matches:
            result.append(f"\n{match['file']}:")
            for line_num, line in match['occurrences'][:5]:  # Show first 5 matches
                result.append(f"  Line {line_num}: {line.strip()}")
                
        return "\n".join(result)
    except Exception as e:
        return f"Error searching code: {str(e)}"


@tool
def call_mcp_tool(tool_name: str, parameters: str) -> str:
    """
    Call an MCP (Model Context Protocol) tool.
    
    Args:
        tool_name: Name of the MCP tool to call
        parameters: JSON string of parameters
        
    Returns:
        Result from MCP tool
    """
    try:
        from config import MCP_ENABLED, MCP_SERVER_URL
        
        if not MCP_ENABLED:
            return "MCP integration is disabled. Set MCP_ENABLED=true to enable."
        
        # Parse parameters
        params = json.loads(parameters) if isinstance(parameters, str) else parameters
        
        # This is a placeholder - actual MCP integration would go here
        # In a real implementation, you would:
        # 1. Connect to MCP server
        # 2. Send tool request
        # 3. Return response
        
        return f"MCP tool '{tool_name}' called with parameters: {params}\n" \
               f"Note: This is a placeholder. Implement actual MCP client integration."
               
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON parameters: {str(e)}"
    except Exception as e:
        return f"Error calling MCP tool: {str(e)}\n{traceback.format_exc()}"


# Collection of all tools
ALL_TOOLS = [
    read_file_tool,
    write_file_tool,
    list_files_tool,
    execute_python_code,
    execute_file,
    analyze_error,
    search_code,
    call_mcp_tool
]


def get_tool_by_name(tool_name: str):
    """
    Get tool by name from the tool collection
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool instance or None if not found
    """
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            return tool
    return None

