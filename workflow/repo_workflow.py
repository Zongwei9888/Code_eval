"""
Autonomous Multi-Agent Repository Analysis System

这是真正的自主代理系统，不是硬编码的workflow！

核心架构：
┌─────────────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS AGENT SYSTEM                                │
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │                    SUPERVISOR NODE                          │      │
│    │   ┌───────────────────────────────────────────────────┐     │      │
│    │   │  1. Observe shared state                          │     │      │
│    │   │  2. LLM reasoning: What should I do next?         │     │      │
│    │   │  3. Decision: {next: "agent", task: "..."}        │     │      │
│    │   └───────────────────────────────────────────────────┘     │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│      ┌───────────┬───────────┼───────────┬───────────┐                  │
│      v           v           v           v           v                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
│  │Planner │ │Research│ │Scanner │ │Analyzer│ │ Fixer  │                │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘                │
│       │          │          │          │          │                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
│  │Executor│ │ Tester │ │Reviewer│ │Environ │ │  Git   │                │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘                │
│       │          │          │          │          │                     │
│       └──────────┴──────────┴──────────┴──────────┘                     │
│                              │                                           │
│                              v                                           │
│                   (Return to SUPERVISOR)                                 │
│                              │                                           │
│              ┌───────────────┴───────────────┐                          │
│              v                               v                          │
│         Continue?                          FINISH                        │
│         (loop back)                         (END)                       │
└─────────────────────────────────────────────────────────────────────────┘

关键特性：
1. Supervisor LLM自主决策下一步
2. 每个Agent有专门的工具集
3. 动态路由，不是硬编码流程
4. 支持反馈循环和策略调整
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm
from agent.state import TrueAgentState, create_true_agent_state
from prompt.supervisor_prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    format_state_summary,
    format_agent_prompt,
    get_available_agents
)
from tools.agent_tools import (
    get_tools_for_agent,
    AGENT_CAPABILITIES
)


# ============================================================================
# SPECIALIST AGENT EXECUTOR
# ============================================================================

def create_specialist_executor(
    llm,
    agent_name: str,
    max_turns: int = 100
):
    """
    创建专家代理执行器
    
    每个代理有自己专属的工具集！
    
    Args:
        llm: LLM实例
        agent_name: 代理名称
        max_turns: 最大工具调用轮数
        
    Returns:
        执行器函数
    """
    # 获取该代理专属的工具
    tools = get_tools_for_agent(agent_name)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    def executor(state: TrueAgentState, task: str) -> Dict[str, Any]:
        """执行代理任务并返回结果"""
        
        # 获取代理专用提示词
        error_context = ""
        if agent_name == "fixer":
            errors = state.get("syntax_errors", []) + state.get("runtime_errors", [])
            if errors:
                error_context = json.dumps(errors[-1], indent=2)
        
        agent_prompt = format_agent_prompt(agent_name, task, {"error_context": error_context})
        
        # 构建消息历史
        messages = [
            SystemMessage(content=agent_prompt)
        ]
        
        # 添加最近上下文
        for msg in state.get("messages", [])[-10:]:
            if isinstance(msg, ToolMessage):
                continue
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            messages.append(msg)
        
        # 添加当前任务
        messages.append(HumanMessage(
            content=f"Task: {task}\n\nProject path: {state['project_path']}"
        ))
        
        # 工具调用循环
        turn = 0
        results = []
        step_logs = []
        
        while turn < max_turns:
            turn += 1
            
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 记录日志
            print(f"\n  [{agent_name.upper()}] Turn {turn}")
            safe_content = content.encode('ascii', 'ignore').decode('ascii')
            print(f"    Response: {safe_content[:150]}...")
            
            # 记录步骤日志（供UI使用）
            step_logs.append({
                "agent": agent_name,
                "turn": turn,
                "type": "llm_response",
                "content": content[:500],
                "has_tool_calls": bool(hasattr(response, 'tool_calls') and response.tool_calls),
                "timestamp": datetime.now().isoformat()
            })
            
            # 检查是否有工具调用
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                results.append({"type": "response", "content": content})
                break
            
            # 执行工具
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 查找并执行工具
                result = f"Tool {tool_name} not found"
                for t in tools:
                    if t.name == tool_name:
                        try:
                            result = t.invoke(tool_args)
                            print(f"    Tool [{tool_name}]: OK")
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            print(f"    Tool [{tool_name}]: ERROR - {str(e)}")
                        break
                
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                results.append({
                    "type": "tool_result",
                    "tool": tool_name,
                    "result": str(result)[:1000]
                })
                
                # 记录工具调用日志
                step_logs.append({
                    "agent": agent_name,
                    "turn": turn,
                    "type": "tool_call",
                    "tool": tool_name,
                    "result": str(result)[:300],
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "agent": agent_name,
            "task": task,
            "results": results,
            "messages": messages[1:],
            "step_logs": step_logs
        }
    
    return executor


# ============================================================================
# AUTONOMOUS AGENT CLASS
# ============================================================================

class AutonomousRepoAgent:
    """
    自主代码仓库分析代理
    
    这是真正的自主代理系统！
    
    核心特性：
    1. Supervisor LLM观察状态并自主决策
    2. 每个专家代理有专属工具集
    3. 动态路由而非硬编码流程
    4. 支持10种专家代理
    """
    
    # 所有可用代理
    AVAILABLE_AGENTS = [
        "planner", "researcher", "scanner", "analyzer", "fixer",
        "executor", "tester", "reviewer", "environment", "git"
    ]
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        max_iterations: int = 20
    ):
        """
        初始化自主代理系统
        
        Args:
            llm_provider: LLM提供商
            max_iterations: 最大迭代次数
        """
        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        
        # 创建LLM实例
        self.supervisor_llm = get_llm(llm_provider, "powerful")  # 用于决策
        self.worker_llm = get_llm(llm_provider, "default")  # 用于执行
        
        # 创建所有专家代理执行器
        self.specialists = {}
        for agent_name in self.AVAILABLE_AGENTS:
            # Fixer使用更强的模型
            llm = self.supervisor_llm if agent_name == "fixer" else self.worker_llm
            max_turns = 8 if agent_name in ["fixer", "researcher"] else 5
            
            self.specialists[agent_name] = create_specialist_executor(
                llm, agent_name, max_turns=max_turns
            )
        
        # 内存检查点
        self.memory = MemorySaver()
        
        # 构建图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        构建代理图
        
        关键：所有代理都返回给Supervisor，由其决定下一步
        """
        workflow = StateGraph(TrueAgentState)
        
        # 添加Supervisor节点
        workflow.add_node("supervisor", self._supervisor_node)
        
        # 添加所有专家代理节点
        for agent_name in self.AVAILABLE_AGENTS:
            node_func = self._create_agent_node(agent_name)
            workflow.add_node(agent_name, node_func)
        
        # 入口点 -> Supervisor
        workflow.add_edge(START, "supervisor")
        
        # Supervisor -> 动态路由（LLM决定！）
        routing_map = {agent: agent for agent in self.AVAILABLE_AGENTS}
        routing_map["FINISH"] = END
        
        workflow.add_conditional_edges(
            "supervisor",
            self._route_by_supervisor,
            routing_map
        )
        
        # 所有专家代理 -> 返回Supervisor（反馈循环）
        for agent_name in self.AVAILABLE_AGENTS:
            workflow.add_edge(agent_name, "supervisor")
        
        return workflow.compile(checkpointer=self.memory)
    
    def _create_agent_node(self, agent_name: str):
        """为指定代理创建节点函数"""
        
        def node(state: TrueAgentState) -> Dict[str, Any]:
            """专家代理节点"""
            task = state.get("current_task", f"Execute {agent_name} task")
            result = self.specialists[agent_name](state, task)
            
            # 解析结果更新状态
            return self._process_agent_result(agent_name, state, result)
        
        return node
    
    def _supervisor_node(self, state: TrueAgentState) -> Dict[str, Any]:
        """
        Supervisor节点 - 系统的大脑
        
        自主决策流程：
        1. 观察当前状态
        2. LLM推理下一步
        3. 返回决策结果
        """
        iteration = state.get("iteration_count", 0) + 1
        max_iter = state.get("max_iterations", self.max_iterations)
        
        print(f"\n{'='*70}")
        print(f"  SUPERVISOR - Iteration {iteration}/{max_iter}")
        print(f"{'='*70}")
        
        # 检查循环限制
        if iteration > max_iter:
            print("  [!] Max iterations reached - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "supervisor_reasoning": "Maximum iterations reached",
                "iteration_count": iteration
            }
        
        # 检查目标是否达成
        if state.get("goal_achieved", False):
            print("  [+] Goal achieved - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "supervisor_reasoning": "Goal has been achieved",
                "iteration_count": iteration
            }
        
        # 格式化状态摘要给LLM
        state_summary = format_state_summary(state)
        
        # 询问Supervisor LLM做决策
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=state_summary)
        ]
        
        response = self.supervisor_llm.invoke(messages)
        content = response.content
        
        print(f"\n  [Supervisor Thinking]")
        safe_content = content.encode('ascii', 'ignore').decode('ascii')
        print(f"    {safe_content[:300]}...")
        
        # 解析决策
        decision = self._parse_supervisor_decision(content)
        
        print(f"\n  [Decision] -> {decision['decision']}")
        print(f"  [Reasoning] {decision['reasoning'][:100]}...")
        
        # 记录决策
        decision_record = {
            "iteration": iteration,
            "decision": decision["decision"],
            "reasoning": decision["reasoning"],
            "task": decision.get("task_for_agent", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        decision_history = list(state.get("decision_history", []))
        decision_history.append(decision_record)
        
        # 更新步骤日志
        step_logs = list(state.get("step_logs", []))
        step_logs.append({
            "agent": "supervisor",
            "action": f"decided: {decision['decision']}",
            "reasoning": decision["reasoning"][:200],
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "supervisor_decision": decision["decision"],
            "supervisor_reasoning": decision["reasoning"],
            "current_task": decision.get("task_for_agent", ""),
            "decision_history": decision_history,
            "iteration_count": iteration,
            "step_logs": step_logs,
            "messages": [response]
        }
    
    def _parse_supervisor_decision(self, content: str) -> Dict[str, Any]:
        """解析Supervisor的决策"""
        
        # 尝试提取JSON
        try:
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 回退：查找关键词
        content_lower = content.lower()
        
        decision = "FINISH"
        reasoning = content[:200]
        
        for agent in self.AVAILABLE_AGENTS:
            if agent in content_lower:
                decision = agent
                break
        
        if "finish" in content_lower or "done" in content_lower or "complete" in content_lower:
            decision = "FINISH"
        
        return {
            "reasoning": reasoning,
            "decision": decision,
            "task_for_agent": content[:300],
            "confidence": "medium"
        }
    
    def _route_by_supervisor(self, state: TrueAgentState) -> str:
        """
        基于Supervisor决策路由
        
        这是与硬编码workflow的关键区别：
        - 旧: if errors -> fixer else -> reporter
        - 新: return state["supervisor_decision"]
        """
        decision = state.get("supervisor_decision", "FINISH")
        valid_decisions = self.AVAILABLE_AGENTS + ["FINISH"]
        
        if decision not in valid_decisions:
            print(f"  [!] Invalid decision '{decision}', defaulting to FINISH")
            return "FINISH"
        
        return decision
    
    def _process_agent_result(
        self, 
        agent_name: str, 
        state: TrueAgentState, 
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理代理结果并更新状态"""
        
        updates = {
            "messages": result.get("messages", []),
            "step_logs": list(state.get("step_logs", [])) + result.get("step_logs", [])
        }
        
        # 根据代理类型处理结果
        if agent_name == "scanner":
            updates.update(self._process_scanner_result(state, result))
        elif agent_name == "analyzer":
            updates.update(self._process_analyzer_result(state, result))
        elif agent_name == "fixer":
            updates.update(self._process_fixer_result(state, result))
        elif agent_name == "executor":
            updates.update(self._process_executor_result(state, result))
        elif agent_name == "tester":
            updates.update(self._process_tester_result(state, result))
        
        # 添加通用日志
        updates["step_logs"].append({
            "agent": agent_name,
            "action": f"completed task",
            "timestamp": datetime.now().isoformat()
        })
        
        return updates
    
    def _process_scanner_result(self, state, result) -> Dict[str, Any]:
        """处理Scanner结果"""
        python_files = []
        test_files = []
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result" and r.get("tool") == "scan_project":
                try:
                    data = json.loads(r.get("result", "{}"))
                    python_files = data.get("python_files", [])
                    test_files = data.get("test_files", [])
                except:
                    pass
        
        return {
            "python_files": python_files,
            "test_files": test_files,
        }
    
    def _process_analyzer_result(self, state, result) -> Dict[str, Any]:
        """处理Analyzer结果"""
        syntax_errors = list(state.get("syntax_errors", []))
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result" and "check" in r.get("tool", ""):
                try:
                    data = json.loads(r.get("result", "{}"))
                    if not data.get("valid", True):
                        syntax_errors.append({
                            "file": data.get("file", "unknown"),
                            "error": data.get("error", "unknown"),
                            "line": data.get("line_number")
                        })
                except:
                    pass
        
        current_file = state.get("current_file")
        if syntax_errors and not current_file:
            current_file = syntax_errors[0].get("file")
        
        return {
            "syntax_errors": syntax_errors,
            "current_file": current_file,
        }
    
    def _process_fixer_result(self, state, result) -> Dict[str, Any]:
        """处理Fixer结果"""
        current_file = state.get("current_file")
        task = state.get("current_task", "")
        
        modifications = list(state.get("modifications", []))
        modifications.append({
            "file": current_file,
            "task": task,
            "agent": "fixer",
            "timestamp": datetime.now().isoformat()
        })
        
        # 检查是否使用了str_replace（正确的方式）
        used_str_replace = any(
            r.get("tool") == "str_replace" 
            for r in result.get("results", []) 
            if r.get("type") == "tool_result"
        )
        
        return {
            "modifications": modifications,
            "used_precise_edit": used_str_replace,
        }
    
    def _process_executor_result(self, state, result) -> Dict[str, Any]:
        """处理Executor结果"""
        current_file = state.get("current_file")
        success = False
        error_message = ""
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result" and "execute" in r.get("tool", ""):
                try:
                    data = json.loads(r.get("result", "{}"))
                    success = data.get("success", False)
                    if not success:
                        error_message = data.get("stderr", "") or data.get("error", "")
                except:
                    pass
        
        execution_history = list(state.get("execution_history", []))
        execution_history.append({
            "file": current_file,
            "success": success,
            "error": error_message[:500],
            "timestamp": datetime.now().isoformat()
        })
        
        runtime_errors = list(state.get("runtime_errors", []))
        if not success and error_message:
            runtime_errors.append({
                "file": current_file,
                "error": error_message[:500],
                "timestamp": datetime.now().isoformat()
            })
        
        # 判断目标是否达成
        goal_achieved = success and not state.get("syntax_errors") and not runtime_errors
        
        return {
            "execution_history": execution_history,
            "last_execution_success": success,
            "last_error_message": error_message,
            "runtime_errors": runtime_errors,
            "goal_achieved": goal_achieved,
        }
    
    def _process_tester_result(self, state, result) -> Dict[str, Any]:
        """处理Tester结果"""
        test_failures = list(state.get("test_failures", []))
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result":
                try:
                    data = json.loads(r.get("result", "{}"))
                    if not data.get("success", True):
                        test_failures.append({
                            "output": data.get("output", "")[:500],
                            "timestamp": datetime.now().isoformat()
                        })
                except:
                    pass
        
        return {
            "test_failures": test_failures,
        }
    
    def run(
        self,
        project_path: str,
        user_request: str = "Analyze the project and fix any code issues",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行自主代理
        
        Args:
            project_path: 项目路径
            user_request: 用户请求
            thread_id: 线程ID（可选）
            
        Returns:
            最终状态
        """
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # 创建初始状态
        initial_state = create_true_agent_state(
            project_path=project_path,
            user_request=user_request,
            max_iterations=self.max_iterations
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'='*70}")
        print(f"  AUTONOMOUS AGENT SYSTEM")
        print(f"{'='*70}")
        print(f"  Project: {project_path}")
        print(f"  Request: {user_request}")
        print(f"  Thread:  {thread_id}")
        print(f"  Max Iterations: {self.max_iterations}")
        print(f"  Available Agents: {len(self.AVAILABLE_AGENTS)}")
        print(f"{'='*70}")
        print(f"\n  Supervisor LLM decides each step dynamically")
        print(f"  Each agent has specialized tools")
        print(f"{'='*70}\n")
        
        # 运行代理
        final_state = self.graph.invoke(initial_state, config)
        
        # 打印摘要
        print(f"\n{'='*70}")
        print(f"  AGENT COMPLETE")
        print(f"{'='*70}")
        print(f"  Iterations: {final_state.get('iteration_count', 0)}")
        print(f"  Decisions made: {len(final_state.get('decision_history', []))}")
        print(f"  Modifications: {len(final_state.get('modifications', []))}")
        print(f"  Goal achieved: {final_state.get('goal_achieved', False)}")
        print(f"{'='*70}\n")
        
        return final_state
    
    def stream_run(
        self,
        project_path: str,
        user_request: str = "Analyze the project and fix any code issues",
        thread_id: Optional[str] = None
    ):
        """
        流式运行代理
        
        Yields:
            状态更新
        """
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        initial_state = create_true_agent_state(
            project_path=project_path,
            user_request=user_request,
            max_iterations=self.max_iterations
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_multi_agent_workflow(
    llm_provider: str = "openrouter",
    max_fix_attempts: int = 20
) -> AutonomousRepoAgent:
    """
    创建多代理工作流的工厂函数
    
    注意：这现在创建的是自主代理，不是硬编码workflow！
    
    Args:
        llm_provider: LLM提供商
        max_fix_attempts: 最大迭代次数
        
    Returns:
        AutonomousRepoAgent实例
    """
    return AutonomousRepoAgent(llm_provider, max_fix_attempts)


# 向后兼容的别名
MultiAgentRepoWorkflow = AutonomousRepoAgent
TrueAgent = AutonomousRepoAgent
create_true_agent = create_multi_agent_workflow
create_agent_executor = create_specialist_executor


# ============================================================================
# EXPORTED NAMES
# ============================================================================

__all__ = [
    "AutonomousRepoAgent",
    "MultiAgentRepoWorkflow",
    "TrueAgent",
    "create_multi_agent_workflow",
    "create_true_agent",
    "create_specialist_executor",
    "create_agent_executor",
]
