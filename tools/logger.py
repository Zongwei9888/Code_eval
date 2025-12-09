"""
Simple Agent Logger - 最小化的日志记录工具

保存每个 agent 的对话记录、工具调用和决策历史
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class AgentLogger:
    """极简 Agent 日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        """初始化日志器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def save_session(self, state: Dict[str, Any], thread_id: str):
        """保存完整会话日志
        
        Args:
            state: agent 最终状态
            thread_id: 会话 ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{thread_id[:8]}.json"
        filepath = self.log_dir / filename
        
        # 构建日志数据
        log_data = {
            "session_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "project_path": state.get("project_path"),
            "user_request": state.get("user_request"),
            "summary": {
                "iterations": state.get("iteration_count", 0),
                "goal_achieved": state.get("goal_achieved", False),
                "total_decisions": len(state.get("decision_history", [])),
                "total_modifications": len(state.get("modifications", [])),
                "execution_attempts": len(state.get("execution_history", [])),
            },
            "decision_history": state.get("decision_history", []),
            "step_logs": state.get("step_logs", []),
            "execution_history": state.get("execution_history", []),
            "modifications": state.get("modifications", []),
            "syntax_errors": state.get("syntax_errors", []),
            "runtime_errors": state.get("runtime_errors", []),
            "error_analysis": state.get("error_analysis", {}),
        }
        
        # 保存为 JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存易读的文本版本
        self._save_readable_log(log_data, filepath.with_suffix(".txt"))
        
        return filepath
    
    def _save_readable_log(self, log_data: Dict[str, Any], filepath: Path):
        """保存易读的文本日志"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"Agent Session Log\n")
            f.write("="*80 + "\n\n")
            
            # 基本信息
            f.write(f"Session ID: {log_data['session_id']}\n")
            f.write(f"Time: {log_data['timestamp']}\n")
            f.write(f"Project: {log_data['project_path']}\n")
            f.write(f"Request: {log_data['user_request']}\n\n")
            
            # 摘要
            summary = log_data['summary']
            f.write("-"*80 + "\n")
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Iterations: {summary['iterations']}\n")
            f.write(f"Goal Achieved: {summary['goal_achieved']}\n")
            f.write(f"Decisions Made: {summary['total_decisions']}\n")
            f.write(f"Code Modifications: {summary['total_modifications']}\n")
            f.write(f"Execution Attempts: {summary['execution_attempts']}\n\n")
            
            # 决策历史
            f.write("-"*80 + "\n")
            f.write("DECISION HISTORY\n")
            f.write("-"*80 + "\n")
            for i, decision in enumerate(log_data['decision_history'], 1):
                f.write(f"\n[{i}] Iteration {decision['iteration']} -> {decision['decision']}\n")
                f.write(f"    Reasoning: {decision['reasoning'][:200]}...\n")
                f.write(f"    Time: {decision['timestamp']}\n")
            
            # Agent 执行日志
            f.write("\n" + "-"*80 + "\n")
            f.write("AGENT EXECUTION LOG\n")
            f.write("-"*80 + "\n")
            for step in log_data['step_logs']:
                agent = step.get('agent', 'unknown').upper()
                action = step.get('action', step.get('type', 'action'))
                time = step.get('timestamp', '')
                
                if step.get('type') == 'llm_response':
                    f.write(f"\n[{agent}] LLM Response\n")
                    f.write(f"    {step.get('content', '')[:150]}...\n")
                elif step.get('type') == 'tool_call':
                    f.write(f"\n[{agent}] Tool: {step.get('tool')}\n")
                    f.write(f"    Result: {step.get('result', '')[:100]}...\n")
                else:
                    f.write(f"\n[{agent}] {action}\n")
                    if 'reasoning' in step:
                        f.write(f"    {step['reasoning'][:150]}...\n")
            
            # 执行历史
            if log_data['execution_history']:
                f.write("\n" + "-"*80 + "\n")
                f.write("EXECUTION HISTORY\n")
                f.write("-"*80 + "\n")
                for i, exec_log in enumerate(log_data['execution_history'], 1):
                    status = "SUCCESS" if exec_log['success'] else "FAILED"
                    f.write(f"\n[{i}] {status} - {exec_log.get('file', 'N/A')}\n")
                    if exec_log['success']:
                        f.write(f"    Output: {exec_log.get('stdout', 'N/A')[:100]}...\n")
                    else:
                        f.write(f"    Error: {exec_log.get('error', 'N/A')[:200]}...\n")
            
            # 错误分析
            if log_data.get('error_analysis'):
                f.write("\n" + "-"*80 + "\n")
                f.write("ERROR ANALYSIS\n")
                f.write("-"*80 + "\n")
                ea = log_data['error_analysis']
                f.write(f"Type: {ea.get('type', 'N/A')}\n")
                f.write(f"Suggested Agent: {ea.get('suggested_agent', 'N/A')}\n")
                f.write(f"Reason: {ea.get('reason', 'N/A')}\n")
                f.write(f"Message: {ea.get('message', 'N/A')}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF LOG\n")
            f.write("="*80 + "\n")


# 全局单例
_logger = None

def get_logger(log_dir: str = "logs") -> AgentLogger:
    """获取全局日志器实例"""
    global _logger
    if _logger is None:
        _logger = AgentLogger(log_dir)
    return _logger

