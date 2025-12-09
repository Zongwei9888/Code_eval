"""
Autonomous Multi-Agent Repository Analysis System

真正的自主代理系统 - 动态决策，非硬编码流程

核心架构（精简版 - 4 Core Agents）：
┌─────────────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS AGENT SYSTEM (4 Agents)                     │
│                                                                          │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │                    SUPERVISOR (Brain)                       │      │
│    │   ┌───────────────────────────────────────────────────┐     │      │
│    │   │  1. Observe State → What's done? What's needed?   │     │      │
│    │   │  2. LLM Reasoning → Analyze errors, suggest fix   │     │      │
│    │   │  3. Decision → Choose next agent OR FINISH        │     │      │
│    │   └───────────────────────────────────────────────────┘     │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│           ┌──────────────────┼──────────────────┐                       │
│           │                  │                  │                       │
│           v                  v                  v                       │
│      ┌────────┐         ┌────────┐        ┌────────┐                   │
│      │Scanner │         │Analyzer│        │ Fixer  │                   │
│      │(Scan)  │         │(Check) │        │(Edit)  │                   │
│      └───┬────┘         └───┬────┘        └───┬────┘                   │
│          │                  │                  │                        │
│          │                  v                  │                        │
│          │             ┌─────────┐             │                        │
│          └────────────>│Executor │<────────────┘                        │
│                        │(Run/Test)│                                     │
│                        └────┬────┘                                      │
│                             │                                           │
│                             v                                           │
│                    ┌────────────────┐                                   │
│                    │  Return State  │                                   │
│                    │  to SUPERVISOR │                                   │
│                    └────────────────┘                                   │
│                             │                                           │
│              ┌──────────────┴──────────────┐                           │
│              v                             v                           │
│      ┌────────────┐                  ┌──────────┐                      │
│      │ Continue?  │                  │ FINISH   │                      │
│      │(Loop back) │                  │  (END)   │                      │
│      └────────────┘                  └──────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘

标准流程：
  Scanner → Analyzer → Fixer (if errors) → Executor → FINISH
                ↑                              ↓
                └──────── (if failed) ─────────┘

关键特性：
1. ✅ Supervisor 自主决策（LLM 推理，非硬编码）
2. ✅ 4 个核心 Agent，专属工具集
3. ✅ 动态路由 + 智能错误分析
4. ✅ 反馈循环（执行失败 → 智能建议 → Fixer 修复）
5. ✅ Memory + Logger（状态持久化 + 完整日志）
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
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
from tools.logger import get_logger


# ============================================================================
# INTELLIGENT ERROR ANALYSIS
# ============================================================================

def analyze_error_and_suggest_agent(error_message: str) -> Tuple[str, str, str]:
    """
    分析错误信息并建议下一个应该调用的 Agent (4 Core Agents)
    
    Core Agents: scanner, analyzer, fixer, executor
    
    Args:
        error_message: 错误信息
        
    Returns:
        Tuple[error_type, suggested_agent, reason]
    """
    error_lower = error_message.lower()
    
    # 1. 依赖/环境问题 -> executor (包含依赖安装功能)
    dependency_patterns = [
        "modulenotfounderror", "no module named",
        "importerror", "cannot import",
        "package not found", "pip install",
        "npm install", "yarn add",
        "cargo add", "go get",
        "missing dependency", "requirements",
        "error: could not find",
        "command not found",
    ]
    if any(p in error_lower for p in dependency_patterns):
        return (
            "dependency_error",
            "executor",
            "Missing dependency. Executor can install packages with run_command or install_dependencies."
        )
    
    # 2. 语法错误 -> fixer
    syntax_patterns = [
        "syntaxerror", "syntax error",
        "indentationerror", "indent",
        "unexpected token", "parsing error",
        "invalid syntax", "unexpected eof",
        "expected", "unexpected",
    ]
    if any(p in error_lower for p in syntax_patterns):
        return (
            "syntax_error",
            "fixer",
            "Syntax error. Fixer can make precise edits to correct the syntax."
        )
    
    # 3. 类型错误 -> fixer
    type_patterns = [
        "typeerror", "type error",
        "attributeerror", "attribute error",
        "has no attribute", "is not callable",
        "cannot read property", "undefined is not",
        "null pointer", "nil pointer",
    ]
    if any(p in error_lower for p in type_patterns):
        return (
            "type_error",
            "fixer",
            "Type/Attribute error. Fixer can correct the code logic."
        )
    
    # 4. 名称错误 -> fixer
    name_patterns = [
        "nameerror", "name error",
        "is not defined", "undefined variable",
        "reference error", "undeclared",
    ]
    if any(p in error_lower for p in name_patterns):
        return (
            "name_error",
            "fixer",
            "Name error - variable or function not defined. Fixer can add definition or fix typo."
        )
    
    # 5. 文件/路径错误 -> analyzer (搜索正确路径)
    file_patterns = [
        "filenotfounderror", "no such file",
        "path not found", "file not found",
        "enoent", "cannot open",
    ]
    if any(p in error_lower for p in file_patterns):
        return (
            "file_error",
            "analyzer",
            "File not found. Analyzer can search for correct file path."
        )
    
    # 6. 权限错误 -> executor
    permission_patterns = [
        "permissionerror", "permission denied",
        "access denied", "eacces",
    ]
    if any(p in error_lower for p in permission_patterns):
        return (
            "permission_error",
            "executor",
            "Permission error. Executor can run chmod or fix permissions."
        )
    
    # 7. 网络/连接错误 -> fixer (通常是代码配置问题)
    network_patterns = [
        "connectionerror", "connection refused",
        "timeout", "network", "socket",
        "econnrefused", "host not found",
    ]
    if any(p in error_lower for p in network_patterns):
        return (
            "network_error",
            "fixer",
            "Network error. Usually a code configuration issue that fixer can address."
        )
    
    # 8. 内存/资源错误 -> fixer
    memory_patterns = [
        "memoryerror", "out of memory",
        "heap", "stack overflow",
        "recursion", "maximum call stack",
    ]
    if any(p in error_lower for p in memory_patterns):
        return (
            "memory_error",
            "fixer",
            "Memory/Stack error. Code may need optimization or recursion fix."
        )
    
    # 9. 测试失败 -> fixer agent
    test_patterns = [
        "assertionerror", "assertion failed",
        "test failed", "expected", "actual",
        "assert", "fail",
    ]
    if any(p in error_lower for p in test_patterns):
        return (
            "test_failure",
            "fixer",
            "Test assertion failed. Fixer can correct the logic to match expected behavior."
        )
    
    # 10. 编译/构建错误
    build_patterns = [
        "compile error", "build failed",
        "linker error", "undefined reference",
        "cargo build", "go build",
        "tsc", "webpack",
    ]
    if any(p in error_lower for p in build_patterns):
        return (
            "build_error",
            "fixer",
            "Build/Compile error. Fixer can address the compilation issues."
        )
    
    # 默认：一般运行时错误 -> fixer
    return (
        "runtime_error",
        "fixer",
        "General runtime error. Fixer agent should analyze and correct the issue."
    )


def get_decision_hints(state: Dict[str, Any]) -> str:
    """
    基于当前状态生成智能决策提示
    
    Args:
        state: 当前状态
        
    Returns:
        决策提示字符串
    """
    hints = []
    
    # 检查项目是否已扫描
    python_files = state.get("python_files", [])
    if not python_files:
        hints.append("📍 Project not scanned yet → Consider calling **scanner** first")
    
    # 检查语法错误
    syntax_errors = state.get("syntax_errors", [])
    if syntax_errors:
        hints.append(f"🔴 {len(syntax_errors)} syntax error(s) pending → **fixer** should fix before execution")
    
    # 检查运行时错误和建议
    error_analysis = state.get("error_analysis", {})
    suggested_agent = state.get("suggested_next_agent")
    if error_analysis and suggested_agent:
        error_type = error_analysis.get("type", "unknown")
        reason = error_analysis.get("reason", "")
        hints.append(f"💡 Error type: {error_type}")
        hints.append(f"   → Suggested: **{suggested_agent}** - {reason}")
    
    # 检查最后执行状态
    execution_history = state.get("execution_history", [])
    if execution_history:
        last_exec = execution_history[-1]
        if last_exec.get("success"):
            hints.append("✅ Last execution succeeded!")
            if not state.get("test_failures"):
                hints.append("   → Consider **FINISH** if goal is achieved")
        else:
            hints.append("❌ Last execution failed - check error analysis above")
    else:
        hints.append("⚠️ Code has NOT been executed yet → **executor** needed before FINISH")
    
    # 检查是否有修改但未验证
    modifications = state.get("modifications", [])
    if modifications and execution_history:
        last_mod_time = modifications[-1].get("timestamp", "")
        last_exec_time = execution_history[-1].get("timestamp", "")
        if last_mod_time > last_exec_time:
            hints.append("🔄 Code modified after last execution → **executor** to verify changes")
    
    # 检查测试失败
    test_failures = state.get("test_failures", [])
    if test_failures:
        hints.append(f"🧪 {len(test_failures)} test failure(s) → **fixer** to fix tests or code")
    
    return "\n".join(hints) if hints else "No specific hints - analyze state and decide"


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
    
    # 核心代理（精简为4个）
    # - scanner: 扫描项目结构
    # - analyzer: 分析代码（包含搜索、语法检查）
    # - fixer: 修复代码错误
    # - executor: 执行代码（包含测试、环境管理）
    AVAILABLE_AGENTS = [
        "scanner",
        "analyzer", 
        "fixer",
        "executor"
    ]
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        max_iterations: int = 100
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
        """为指定代理创建节点函数 - 支持动态交互"""
        
        def node(state: TrueAgentState) -> Dict[str, Any]:
            """
            专家代理节点 - 动态交互版本
            
            1. 接收来自 Supervisor 的任务和上下文
            2. 执行任务并生成结构化输出
            3. 生成反馈给 Supervisor
            4. 准备上下文给下一个 agent
            """
            # 获取当前任务（来自 Supervisor）
            task = state.get("current_task", f"Execute {agent_name} task")
            
            # 获取传递的上下文（来自上一个 agent）
            context_from_prev = state.get("context_for_next_agent", {})
            
            print(f"\n  [{agent_name.upper()}] Received task: {task[:100]}...")
            if context_from_prev:
                print(f"  [{agent_name.upper()}] Context from previous agent: {list(context_from_prev.keys())}")
            
            # 执行代理任务
            result = self.specialists[agent_name](state, task)
            
            # 解析结果并生成反馈
            updates = self._process_agent_result(agent_name, state, result)
            
            # 生成反馈给 Supervisor（描述 agent 做了什么，发现了什么）
            feedback = self._generate_agent_feedback(agent_name, state, result, updates)
            updates["agent_feedback"] = feedback
            updates["last_agent"] = agent_name
            updates["last_agent_output"] = self._extract_structured_output(agent_name, result, updates)
            
            # 准备上下文给下一个 agent
            context_for_next = self._prepare_context_for_next_agent(agent_name, state, updates)
            updates["context_for_next_agent"] = context_for_next
            
            print(f"  [{agent_name.upper()}] Feedback: {feedback[:150]}...")
            
            return updates
        
        return node
    
    def _supervisor_node(self, state: TrueAgentState) -> Dict[str, Any]:
        """
        Supervisor节点 - 系统的大脑
        
        自主决策流程：
        1. 观察当前状态
        2. LLM推理下一步
        3. 返回决策结果
        
        关键：必须执行验证才能结束！
        """
        iteration = state.get("iteration_count", 0) + 1
        max_iter = state.get("max_iterations", self.max_iterations)
        execution_history = state.get("execution_history", [])
        has_executed = len(execution_history) > 0
        python_files = state.get("python_files", [])
        syntax_errors = state.get("syntax_errors", [])
        
        print(f"\n{'='*70}")
        print(f"  SUPERVISOR - Iteration {iteration}/{max_iter}")
        print(f"  Python files: {len(python_files)}, Syntax errors: {len(syntax_errors)}")
        print(f"  Execution attempts: {len(execution_history)}")
        print(f"{'='*70}")
        
        # 检查循环限制
        if iteration > max_iter:
            print("  [!] Max iterations reached - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "supervisor_reasoning": "Maximum iterations reached",
                "iteration_count": iteration
            }
        
        # 检查目标是否达成 - 必须有执行验证！
        if state.get("goal_achieved", False) and has_executed:
            print("  [+] Goal achieved with execution verification - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "supervisor_reasoning": "Goal has been achieved and code execution verified",
                "iteration_count": iteration
            }
        
        # 如果没有执行但goal_achieved为True，强制执行验证
        if state.get("goal_achieved", False) and not has_executed:
            print("  [!] Goal marked achieved but NO execution - forcing executor")
            return {
                "supervisor_decision": "executor",
                "supervisor_reasoning": "Code has not been executed yet - must verify before finish",
                "current_task": "Execute the main Python file to verify the code works correctly",
                "iteration_count": iteration,
                "step_logs": list(state.get("step_logs", [])) + [{
                    "agent": "supervisor",
                    "action": "forcing executor - no execution yet",
                    "timestamp": datetime.now().isoformat()
                }]
            }
        
        # 检查是否陷入循环（同一个 agent 连续调用 3 次）
        decision_history = state.get("decision_history", [])
        if len(decision_history) >= 3:
            last_3_decisions = [d["decision"] for d in decision_history[-3:]]
            if len(set(last_3_decisions)) == 1:  # 三次决策都是同一个 agent
                agent = last_3_decisions[0]
                print(f"  [!] LOOP DETECTED: {agent} called 3 times in a row!")
                
                # 强制打破循环
                if agent == "scanner" and python_files:
                    print(f"  [!] Breaking loop: scanner -> analyzer (files exist)")
                    return {
                        "supervisor_decision": "analyzer",
                        "supervisor_reasoning": "Breaking scanner loop - files already discovered, moving to analysis",
                        "current_task": "Analyze the discovered Python files and check for syntax errors",
                        "iteration_count": iteration
                    }
                elif agent == "analyzer" and not syntax_errors:
                    print(f"  [!] Breaking loop: analyzer -> executor (no errors)")
                    return {
                        "supervisor_decision": "executor",
                        "supervisor_reasoning": "Breaking analyzer loop - no syntax errors found, proceed to execution",
                        "current_task": "Execute the main Python file to verify functionality",
                        "iteration_count": iteration
                    }
                elif agent == "fixer":
                    print(f"  [!] Breaking loop: fixer -> executor (test the fix)")
                    return {
                        "supervisor_decision": "executor",
                        "supervisor_reasoning": "Breaking fixer loop - attempting execution to verify fixes",
                        "current_task": "Execute code to check if fixes resolved the issues",
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
        
        关键安全检查：如果要FINISH但没执行过代码，强制转到executor
        """
        decision = state.get("supervisor_decision", "FINISH")
        valid_decisions = self.AVAILABLE_AGENTS + ["FINISH"]
        
        if decision not in valid_decisions:
            print(f"  [!] Invalid decision '{decision}', defaulting to FINISH")
            return "FINISH"
        
        # 安全检查：如果决定FINISH但从未执行过代码，强制执行
        execution_history = state.get("execution_history", [])
        if decision == "FINISH" and len(execution_history) == 0:
            python_files = state.get("python_files", [])
            if python_files:  # 如果有Python文件可执行
                print(f"  [!] Overriding FINISH -> executor (no execution history)")
                return "executor"
        
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
        """处理Scanner结果 - 解析 scan_project 返回的项目文件信息"""
        python_files = []
        test_files = []
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result" and r.get("tool") == "scan_project":
                try:
                    data = json.loads(r.get("result", "{}"))
                    # scan_project 返回格式：{"python_files": [...], "test_files": [...], ...}
                    python_files = data.get("python_files", [])
                    test_files = data.get("test_files", [])
                    
                    # 打印调试信息
                    print(f"    [Scanner] Found {len(python_files)} Python files, {len(test_files)} test files")
                    if python_files:
                        print(f"    [Scanner] Sample files: {python_files[:3]}")
                except Exception as e:
                    print(f"    [Scanner] Error parsing result: {e}")
        
        # 确保不返回 None 或空列表问题
        if not python_files:
            print(f"    [Scanner] WARNING: No Python files detected!")
        
        return {
            "python_files": python_files if python_files else [],
            "test_files": test_files if test_files else [],
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
        """
        处理Executor结果
        
        智能分析错误并建议下一个 Agent
        """
        current_file = state.get("current_file")
        success = False
        error_message = ""
        stdout_message = ""
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result":
                tool_name = r.get("tool", "")
                if "execute" in tool_name or "run" in tool_name:
                    try:
                        data = json.loads(r.get("result", "{}"))
                        success = data.get("success", False)
                        stdout_message = data.get("stdout", "")
                        if not success:
                            error_message = data.get("stderr", "") or data.get("error", "")
                    except:
                        pass
        
        execution_history = list(state.get("execution_history", []))
        execution_history.append({
            "file": current_file,
            "success": success,
            "error": error_message[:500],
            "stdout": stdout_message[:200] if stdout_message else "",
            "timestamp": datetime.now().isoformat()
        })
        
        runtime_errors = list(state.get("runtime_errors", []))
        
        # 智能错误分析
        error_analysis = {}
        suggested_next_agent = None
        
        if not success and error_message:
            # 使用智能错误分析
            error_type, suggested_agent, reason = analyze_error_and_suggest_agent(error_message)
            
            error_analysis = {
                "type": error_type,
                "message": error_message[:300],
                "suggested_agent": suggested_agent,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            suggested_next_agent = suggested_agent
            
            runtime_errors.append({
                "file": current_file,
                "error": error_message[:500],
                "error_type": error_type,
                "suggested_agent": suggested_agent,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"    [Error Analysis] Type: {error_type}")
            print(f"    [Suggested Agent] {suggested_agent} - {reason}")
        
        # 判断目标是否达成
        goal_achieved = success and not state.get("syntax_errors") and len(runtime_errors) == 0
        
        return {
            "execution_history": execution_history,
            "last_execution_success": success,
            "last_error_message": error_message,
            "runtime_errors": runtime_errors,
            "goal_achieved": goal_achieved,
            "error_analysis": error_analysis,
            "suggested_next_agent": suggested_next_agent,
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
    
    def _generate_agent_feedback(
        self,
        agent_name: str,
        state: TrueAgentState,
        result: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> str:
        """
        生成 agent 反馈给 Supervisor
        
        这是动态交互的核心 - agent 告诉 Supervisor 它做了什么，发现了什么
        """
        feedback_parts = []
        
        if agent_name == "scanner":
            python_files = updates.get("python_files", [])
            test_files = updates.get("test_files", [])
            if python_files:
                feedback_parts.append(f"Scanned project successfully.")
                feedback_parts.append(f"Found {len(python_files)} Python files and {len(test_files)} test files.")
                feedback_parts.append(f"Ready for code analysis.")
            else:
                feedback_parts.append("Scan completed but no Python files found.")
                
        elif agent_name == "analyzer":
            syntax_errors = updates.get("syntax_errors", [])
            if syntax_errors:
                feedback_parts.append(f"Analysis complete: Found {len(syntax_errors)} syntax error(s).")
                feedback_parts.append(f"File with errors: {syntax_errors[0].get('file', 'unknown')}")
                feedback_parts.append(f"Recommendation: Call fixer to resolve errors.")
            else:
                feedback_parts.append("Analysis complete: No syntax errors detected.")
                feedback_parts.append("Code appears syntactically correct.")
                feedback_parts.append("Recommendation: Proceed to execution.")
                
        elif agent_name == "fixer":
            modifications = updates.get("modifications", [])
            if modifications:
                last_mod = modifications[-1]
                feedback_parts.append(f"Fix applied to {last_mod.get('file', 'file')}.")
                feedback_parts.append(f"Used precise edit tools.")
                feedback_parts.append("Recommendation: Run executor to verify the fix.")
            else:
                feedback_parts.append("Fixer ran but no modifications made.")
                
        elif agent_name == "executor":
            execution_history = updates.get("execution_history", [])
            if execution_history:
                last_exec = execution_history[-1]
                if last_exec.get("success"):
                    feedback_parts.append("Execution successful!")
                    feedback_parts.append(f"Output: {last_exec.get('stdout', 'N/A')[:100]}")
                    feedback_parts.append("All checks passed. Ready to finish.")
                else:
                    error = last_exec.get("error", "Unknown error")
                    feedback_parts.append(f"Execution failed: {error[:150]}")
                    
                    # 使用智能错误分析
                    error_analysis = updates.get("error_analysis", {})
                    if error_analysis:
                        suggested = error_analysis.get("suggested_agent", "fixer")
                        reason = error_analysis.get("reason", "")
                        feedback_parts.append(f"Error type: {error_analysis.get('type', 'unknown')}")
                        feedback_parts.append(f"Recommendation: Call {suggested} - {reason}")
            else:
                feedback_parts.append("Executor ran but no execution recorded.")
        
        return " ".join(feedback_parts) if feedback_parts else f"{agent_name} completed its task."
    
    def _extract_structured_output(
        self,
        agent_name: str,
        result: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        从 agent 结果中提取结构化输出
        
        这让 Supervisor 能够程序化地理解 agent 的输出
        """
        output = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        if agent_name == "scanner":
            output["python_files_count"] = len(updates.get("python_files", []))
            output["test_files_count"] = len(updates.get("test_files", []))
            output["scan_successful"] = output["python_files_count"] > 0
            
        elif agent_name == "analyzer":
            syntax_errors = updates.get("syntax_errors", [])
            output["errors_found"] = len(syntax_errors)
            output["has_errors"] = len(syntax_errors) > 0
            if syntax_errors:
                output["first_error_file"] = syntax_errors[0].get("file")
                output["first_error_msg"] = syntax_errors[0].get("error")
                
        elif agent_name == "fixer":
            modifications = updates.get("modifications", [])
            output["modifications_count"] = len(modifications)
            output["used_precise_edit"] = updates.get("used_precise_edit", False)
            if modifications:
                output["last_modified_file"] = modifications[-1].get("file")
                
        elif agent_name == "executor":
            execution_history = updates.get("execution_history", [])
            if execution_history:
                last_exec = execution_history[-1]
                output["execution_success"] = last_exec.get("success", False)
                output["exit_code"] = 0 if last_exec.get("success") else 1
                output["has_output"] = bool(last_exec.get("stdout"))
                output["has_error"] = bool(last_exec.get("error"))
                
                if not last_exec.get("success"):
                    error_analysis = updates.get("error_analysis", {})
                    output["error_type"] = error_analysis.get("type", "unknown")
                    output["suggested_next_agent"] = error_analysis.get("suggested_agent", "fixer")
        
        return output
    
    def _prepare_context_for_next_agent(
        self,
        agent_name: str,
        state: TrueAgentState,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        准备传递给下一个 agent 的上下文
        
        这让 agent 之间可以共享关键信息
        """
        context = {}
        
        if agent_name == "scanner":
            # Scanner -> Analyzer: 传递文件列表
            python_files = updates.get("python_files", [])
            if python_files:
                context["files_to_analyze"] = python_files[:5]  # 前5个文件
                context["total_files"] = len(python_files)
                
        elif agent_name == "analyzer":
            # Analyzer -> Fixer: 传递错误信息
            syntax_errors = updates.get("syntax_errors", [])
            if syntax_errors:
                context["errors_to_fix"] = syntax_errors
                context["priority_file"] = syntax_errors[0].get("file")
                context["priority_error"] = syntax_errors[0].get("error")
                
        elif agent_name == "fixer":
            # Fixer -> Executor: 传递修改的文件
            modifications = updates.get("modifications", [])
            if modifications:
                last_mod = modifications[-1]
                context["modified_file"] = last_mod.get("file")
                context["modification_type"] = last_mod.get("task")
                context["should_verify"] = True
                
        elif agent_name == "executor":
            # Executor -> Fixer: 传递执行错误
            execution_history = updates.get("execution_history", [])
            if execution_history:
                last_exec = execution_history[-1]
                if not last_exec.get("success"):
                    context["execution_error"] = last_exec.get("error", "")
                    context["failed_file"] = last_exec.get("file")
                    
                    error_analysis = updates.get("error_analysis", {})
                    context["error_type"] = error_analysis.get("type", "unknown")
                    context["fix_suggestion"] = error_analysis.get("reason", "")
        
        return context
    
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
        
        # 自动保存日志
        try:
            logger = get_logger()
            log_file = logger.save_session(final_state, thread_id)
            print(f"  Log saved: {log_file}")
        except Exception as e:
            print(f"  Warning: Failed to save log - {e}")
        
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
    max_fix_attempts: int = 20,
    max_iterations: int = None
) -> AutonomousRepoAgent:
    """
    创建多代理工作流的工厂函数
    
    注意：这现在创建的是自主代理，不是硬编码workflow！
    
    Args:
        llm_provider: LLM提供商
        max_fix_attempts: 最大迭代次数 (deprecated, use max_iterations)
        max_iterations: 最大迭代次数 (preferred)
        
    Returns:
        AutonomousRepoAgent实例
    """
    # 优先使用 max_iterations，向后兼容 max_fix_attempts
    iterations = max_iterations if max_iterations is not None else max_fix_attempts
    return AutonomousRepoAgent(llm_provider, iterations)


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
