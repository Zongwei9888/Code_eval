"""
Git Integration Tools

Git 版本控制集成工具，用于查看状态、变更、提交和回滚。
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool


def _run_git_command(args: List[str], cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
    """执行Git命令的辅助函数"""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "命令超时"}
    except FileNotFoundError:
        return {"success": False, "error": "Git未安装或不在PATH中"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def git_status(project_path: str = ".") -> str:
    """
    获取Git仓库状态。
    
    显示工作区和暂存区的文件状态。
    
    Args:
        project_path: 项目路径
        
    Returns:
        Git状态信息
        
    Example:
        git_status("./my-project")
    """
    result = _run_git_command(["status", "--porcelain", "-b"], cwd=project_path)
    
    if not result["success"]:
        return json.dumps({
            "error": result.get("error") or result.get("stderr", "未知错误"),
            "is_git_repo": False
        })
    
    output = result["stdout"]
    lines = output.strip().split('\n') if output.strip() else []
    
    # 解析状态
    branch = None
    staged = []
    unstaged = []
    untracked = []
    
    for line in lines:
        if line.startswith("##"):
            # 分支信息
            branch = line[3:].split("...")[0]
        elif line.startswith("A "):
            staged.append({"file": line[3:], "status": "added"})
        elif line.startswith("M "):
            staged.append({"file": line[3:], "status": "modified"})
        elif line.startswith("D "):
            staged.append({"file": line[3:], "status": "deleted"})
        elif line.startswith(" M"):
            unstaged.append({"file": line[3:], "status": "modified"})
        elif line.startswith(" D"):
            unstaged.append({"file": line[3:], "status": "deleted"})
        elif line.startswith("??"):
            untracked.append(line[3:])
        elif line.startswith("MM"):
            staged.append({"file": line[3:], "status": "modified"})
            unstaged.append({"file": line[3:], "status": "modified"})
    
    return json.dumps({
        "is_git_repo": True,
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0
    }, ensure_ascii=False, indent=2)


@tool
def git_diff(
    project_path: str = ".",
    file_path: str = None,
    staged: bool = False,
    commit: str = None
) -> str:
    """
    查看Git差异。
    
    Args:
        project_path: 项目路径
        file_path: 可选，只查看指定文件的差异
        staged: 是否查看暂存区的差异
        commit: 可选，与指定提交比较
        
    Returns:
        Diff内容
        
    Example:
        # 查看所有未暂存的变更
        git_diff("./my-project")
        
        # 查看暂存区的变更
        git_diff("./my-project", staged=True)
        
        # 查看特定文件的变更
        git_diff("./my-project", file_path="main.py")
    """
    args = ["diff"]
    
    if staged:
        args.append("--staged")
    
    if commit:
        args.append(commit)
    
    if file_path:
        args.extend(["--", file_path])
    
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"error": result.get("error") or result.get("stderr")})
    
    diff_content = result["stdout"]
    
    if not diff_content.strip():
        return json.dumps({
            "has_changes": False,
            "message": "没有差异"
        })
    
    return json.dumps({
        "has_changes": True,
        "diff": diff_content[:10000]  # 限制长度
    }, ensure_ascii=False)


@tool
def git_log(
    project_path: str = ".",
    max_commits: int = 10,
    file_path: str = None,
    oneline: bool = True
) -> str:
    """
    查看Git提交历史。
    
    Args:
        project_path: 项目路径
        max_commits: 最大显示提交数
        file_path: 可选，只查看指定文件的历史
        oneline: 是否使用单行格式
        
    Returns:
        提交历史
    """
    args = ["log", f"-{max_commits}"]
    
    if oneline:
        args.append("--oneline")
    else:
        args.extend(["--pretty=format:%H|%an|%ad|%s", "--date=short"])
    
    if file_path:
        args.extend(["--", file_path])
    
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"error": result.get("error") or result.get("stderr")})
    
    output = result["stdout"]
    
    if oneline:
        commits = []
        for line in output.strip().split('\n'):
            if line:
                parts = line.split(' ', 1)
                commits.append({
                    "hash": parts[0],
                    "message": parts[1] if len(parts) > 1 else ""
                })
    else:
        commits = []
        for line in output.strip().split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) >= 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3]
                    })
    
    return json.dumps({
        "commits": commits,
        "count": len(commits)
    }, ensure_ascii=False, indent=2)


@tool
def git_show(
    project_path: str = ".",
    commit: str = "HEAD",
    file_path: str = None
) -> str:
    """
    显示特定提交的内容。
    
    Args:
        project_path: 项目路径
        commit: 提交哈希或引用（默认HEAD）
        file_path: 可选，只显示指定文件在该提交时的内容
        
    Returns:
        提交内容或文件内容
    """
    if file_path:
        args = ["show", f"{commit}:{file_path}"]
    else:
        args = ["show", commit, "--stat"]
    
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"error": result.get("error") or result.get("stderr")})
    
    return json.dumps({
        "commit": commit,
        "content": result["stdout"][:10000]
    }, ensure_ascii=False)


@tool
def git_add(
    project_path: str = ".",
    files: str = "."
) -> str:
    """
    将文件添加到暂存区。
    
    Args:
        project_path: 项目路径
        files: 要添加的文件（"." 表示全部，或用空格分隔多个文件）
        
    Returns:
        操作结果
    """
    args = ["add"] + files.split()
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"success": False, "error": result.get("stderr")})
    
    return json.dumps({
        "success": True,
        "message": f"已添加: {files}"
    })


@tool
def git_commit(
    project_path: str = ".",
    message: str = None
) -> str:
    """
    提交暂存的变更。
    
    Args:
        project_path: 项目路径
        message: 提交信息（必填）
        
    Returns:
        提交结果
        
    IMPORTANT: 在提交前请确保已使用 git_add 添加文件到暂存区
    """
    if not message:
        return json.dumps({"success": False, "error": "提交信息不能为空"})
    
    result = _run_git_command(["commit", "-m", message], cwd=project_path)
    
    if not result["success"]:
        stderr = result.get("stderr", "")
        if "nothing to commit" in stderr:
            return json.dumps({
                "success": False,
                "error": "没有可提交的变更，请先使用 git_add 添加文件"
            })
        return json.dumps({"success": False, "error": stderr})
    
    return json.dumps({
        "success": True,
        "message": "提交成功",
        "output": result["stdout"]
    })


@tool
def git_revert_file(
    project_path: str = ".",
    file_path: str = None
) -> str:
    """
    回滚单个文件的未暂存变更。
    
    WARNING: 此操作会丢失未保存的更改！
    
    Args:
        project_path: 项目路径
        file_path: 要回滚的文件路径
        
    Returns:
        回滚结果
    """
    if not file_path:
        return json.dumps({"success": False, "error": "请指定要回滚的文件"})
    
    result = _run_git_command(["checkout", "--", file_path], cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"success": False, "error": result.get("stderr")})
    
    return json.dumps({
        "success": True,
        "message": f"已回滚文件: {file_path}"
    })


@tool
def git_stash(
    project_path: str = ".",
    action: str = "push",
    message: str = None
) -> str:
    """
    Git stash 操作。
    
    Args:
        project_path: 项目路径
        action: "push"（保存）, "pop"（恢复并删除）, "list"（列出）, "apply"（恢复不删除）
        message: stash消息（仅push时使用）
        
    Returns:
        操作结果
    """
    if action == "push":
        args = ["stash", "push"]
        if message:
            args.extend(["-m", message])
    elif action in ["pop", "list", "apply"]:
        args = ["stash", action]
    else:
        return json.dumps({"success": False, "error": f"无效的action: {action}"})
    
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"success": False, "error": result.get("stderr")})
    
    return json.dumps({
        "success": True,
        "action": action,
        "output": result["stdout"]
    })


@tool
def git_branch(
    project_path: str = ".",
    action: str = "list",
    branch_name: str = None
) -> str:
    """
    Git分支操作。
    
    Args:
        project_path: 项目路径
        action: "list"（列出）, "create"（创建）, "switch"（切换）, "delete"（删除）
        branch_name: 分支名称（create/switch/delete时需要）
        
    Returns:
        操作结果
    """
    if action == "list":
        args = ["branch", "-a"]
    elif action == "create":
        if not branch_name:
            return json.dumps({"success": False, "error": "请指定分支名称"})
        args = ["branch", branch_name]
    elif action == "switch":
        if not branch_name:
            return json.dumps({"success": False, "error": "请指定分支名称"})
        args = ["checkout", branch_name]
    elif action == "delete":
        if not branch_name:
            return json.dumps({"success": False, "error": "请指定分支名称"})
        args = ["branch", "-d", branch_name]
    else:
        return json.dumps({"success": False, "error": f"无效的action: {action}"})
    
    result = _run_git_command(args, cwd=project_path)
    
    if not result["success"]:
        return json.dumps({"success": False, "error": result.get("stderr")})
    
    if action == "list":
        branches = []
        current = None
        for line in result["stdout"].strip().split('\n'):
            if line.startswith('*'):
                current = line[2:].strip()
                branches.append(current)
            else:
                branches.append(line.strip())
        return json.dumps({
            "branches": branches,
            "current": current
        }, ensure_ascii=False, indent=2)
    
    return json.dumps({
        "success": True,
        "action": action,
        "branch": branch_name
    })


# ============================================================================
# TOOL COLLECTION
# ============================================================================

GIT_TOOLS = [
    git_status,
    git_diff,
    git_log,
    git_show,
    git_add,
    git_commit,
    git_revert_file,
    git_stash,
    git_branch
]

