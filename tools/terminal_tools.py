"""
Terminal Command Tools

终端命令执行工具，支持前台和后台运行。

注意：这些工具目前在本地执行。生产环境应该使用沙箱（如E2B、Docker）。
"""

import os
import sys
import subprocess
import signal
import json
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.tools import tool


# ============================================================================
# PROCESS MANAGER
# ============================================================================

@dataclass
class ProcessInfo:
    """进程信息"""
    pid: int
    command: str
    cwd: str
    started_at: str
    process: subprocess.Popen
    output_queue: queue.Queue = field(default_factory=queue.Queue)


class ProcessManager:
    """后台进程管理器"""
    
    def __init__(self):
        self.processes: Dict[int, ProcessInfo] = {}
        self._lock = threading.Lock()
    
    def add(self, info: ProcessInfo):
        with self._lock:
            self.processes[info.pid] = info
    
    def remove(self, pid: int) -> Optional[ProcessInfo]:
        with self._lock:
            return self.processes.pop(pid, None)
    
    def get(self, pid: int) -> Optional[ProcessInfo]:
        with self._lock:
            return self.processes.get(pid)
    
    def list_all(self) -> List[Dict]:
        with self._lock:
            result = []
            for pid, info in self.processes.items():
                is_running = info.process.poll() is None
                result.append({
                    "pid": pid,
                    "command": info.command,
                    "cwd": info.cwd,
                    "started_at": info.started_at,
                    "is_running": is_running,
                    "exit_code": info.process.returncode if not is_running else None
                })
            return result


# 全局进程管理器
_process_manager = ProcessManager()


# ============================================================================
# COMMAND EXECUTION TOOLS
# ============================================================================

@tool
def run_command(
    command: str,
    cwd: str = None,
    timeout: int = 60,
    env: str = None
) -> str:
    """
    执行shell命令并等待完成。
    
    适用于：
    - 安装依赖 (pip install, npm install)
    - 运行测试 (pytest, npm test)
    - 编译构建 (make, cargo build)
    - 文件操作 (mkdir, cp, mv)
    
    Args:
        command: 要执行的命令
        cwd: 工作目录（默认当前目录）
        timeout: 超时时间（秒）
        env: 额外的环境变量（JSON格式）
        
    Returns:
        命令输出和执行状态
        
    Example:
        # 安装Python依赖
        run_command("pip install requests")
        
        # 在指定目录运行测试
        run_command("pytest -v", cwd="./tests")
        
        # 带环境变量
        run_command("python main.py", env='{"DEBUG": "true"}')
        
    WARNING: 此工具在本地执行命令。生产环境应使用沙箱。
    """
    try:
        # 解析环境变量
        extra_env = {}
        if env:
            try:
                extra_env = json.loads(env)
            except json.JSONDecodeError:
                return json.dumps({"success": False, "error": "无效的env JSON格式"})
        
        # 合并环境变量
        full_env = os.environ.copy()
        full_env.update(extra_env)
        
        # 确定工作目录
        work_dir = cwd if cwd else os.getcwd()
        if not Path(work_dir).exists():
            return json.dumps({"success": False, "error": f"目录不存在: {work_dir}"})
        
        # 执行命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env=full_env
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[-5000:] if result.stdout else "",  # 限制输出长度
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "command": command,
            "cwd": work_dir
        }, ensure_ascii=False, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": f"命令超时 ({timeout}秒)",
            "command": command
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "command": command
        })


@tool
def run_background(
    command: str,
    cwd: str = None
) -> str:
    """
    在后台运行命令（不等待完成）。
    
    适用于：
    - 启动开发服务器
    - 运行长时间任务
    - 启动监控进程
    
    Args:
        command: 要执行的命令
        cwd: 工作目录
        
    Returns:
        进程ID（用于后续管理）
        
    Example:
        # 启动开发服务器
        run_background("python -m http.server 8000")
        
        # 启动Node服务
        run_background("npm run dev", cwd="./frontend")
        
    NOTE: 使用 kill_process 终止后台进程，使用 list_processes 查看所有进程
    """
    try:
        work_dir = cwd if cwd else os.getcwd()
        if not Path(work_dir).exists():
            return json.dumps({"success": False, "error": f"目录不存在: {work_dir}"})
        
        # 启动进程
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            text=True
        )
        
        # 记录进程信息
        info = ProcessInfo(
            pid=process.pid,
            command=command,
            cwd=work_dir,
            started_at=datetime.now().isoformat(),
            process=process
        )
        _process_manager.add(info)
        
        # 启动输出收集线程
        def collect_output(proc, q):
            try:
                for line in proc.stdout:
                    q.put(("stdout", line))
                for line in proc.stderr:
                    q.put(("stderr", line))
            except:
                pass
        
        thread = threading.Thread(target=collect_output, args=(process, info.output_queue))
        thread.daemon = True
        thread.start()
        
        return json.dumps({
            "success": True,
            "pid": process.pid,
            "command": command,
            "cwd": work_dir,
            "message": f"进程已启动，PID: {process.pid}"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "command": command
        })


@tool
def kill_process(pid: int) -> str:
    """
    终止后台进程。
    
    Args:
        pid: 进程ID（从 run_background 获取）
        
    Returns:
        终止结果
    """
    try:
        info = _process_manager.get(pid)
        
        if not info:
            return json.dumps({
                "success": False,
                "error": f"未找到进程 {pid}（可能不是由此工具启动的）"
            })
        
        # 终止进程
        if info.process.poll() is None:  # 进程仍在运行
            info.process.terminate()
            try:
                info.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                info.process.kill()
                info.process.wait()
        
        _process_manager.remove(pid)
        
        return json.dumps({
            "success": True,
            "pid": pid,
            "message": f"进程 {pid} 已终止"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def list_processes() -> str:
    """
    列出所有由 run_background 启动的后台进程。
    
    Returns:
        进程列表
    """
    processes = _process_manager.list_all()
    
    return json.dumps({
        "processes": processes,
        "count": len(processes)
    }, ensure_ascii=False, indent=2)


@tool
def get_process_output(
    pid: int,
    max_lines: int = 100
) -> str:
    """
    获取后台进程的输出。
    
    Args:
        pid: 进程ID
        max_lines: 最大返回行数
        
    Returns:
        进程输出
    """
    info = _process_manager.get(pid)
    
    if not info:
        return json.dumps({
            "success": False,
            "error": f"未找到进程 {pid}"
        })
    
    # 收集队列中的输出
    lines = []
    try:
        while not info.output_queue.empty() and len(lines) < max_lines:
            stream, line = info.output_queue.get_nowait()
            lines.append({"stream": stream, "line": line.rstrip()})
    except:
        pass
    
    is_running = info.process.poll() is None
    
    return json.dumps({
        "pid": pid,
        "is_running": is_running,
        "exit_code": info.process.returncode if not is_running else None,
        "output_lines": lines,
        "lines_returned": len(lines)
    }, ensure_ascii=False, indent=2)


@tool
def check_port(port: int) -> str:
    """
    检查端口是否被占用。
    
    Args:
        port: 端口号
        
    Returns:
        端口状态
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            
            if result == 0:
                return json.dumps({
                    "port": port,
                    "status": "in_use",
                    "message": f"端口 {port} 正在使用中"
                })
            else:
                return json.dumps({
                    "port": port,
                    "status": "available",
                    "message": f"端口 {port} 可用"
                })
    except Exception as e:
        return json.dumps({
            "port": port,
            "status": "error",
            "error": str(e)
        })


@tool
def get_system_info() -> str:
    """
    获取系统信息。
    
    Returns:
        系统信息（OS、Python版本等）
    """
    import platform
    
    return json.dumps({
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "home": str(Path.home())
    }, ensure_ascii=False, indent=2)


# ============================================================================
# TOOL COLLECTION
# ============================================================================

TERMINAL_TOOLS = [
    run_command,
    run_background,
    kill_process,
    list_processes,
    get_process_output,
    check_port,
    get_system_info
]

