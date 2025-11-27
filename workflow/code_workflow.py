"""
Autonomous Single-File Code Agent

单文件代码分析的自主代理系统

与repo_workflow.py类似，但专注于单个文件的分析和修复。
使用Supervisor自主决策，而不是硬编码workflow。

架构：
┌─────────────────────────────────────────────────────────────────┐
│                   SINGLE FILE AGENT                              │
│                                                                  │
│    ┌───────────────────────────────────────────────────────┐    │
│    │                  SUPERVISOR                           │    │
│    │   观察状态 -> LLM推理 -> 选择下一步                    │    │
│    └───────────────────────────────────────────────────────┘    │
│                              │                                   │
│           ┌──────────────────┼──────────────────┐               │
│           v                  v                  v               │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│     │ Analyzer │      │  Fixer   │      │ Executor │           │
│     └────┬─────┘      └────┬─────┘      └────┬─────┘           │
│          │                 │                 │                  │
│          └─────────────────┴─────────────────┘                  │
│                            │                                     │
│                            v                                     │
│                   Return to SUPERVISOR                           │
│                            │                                     │
│                ┌───────────┴───────────┐                        │
│                v                       v                        │
│           Continue?                 FINISH                       │
└─────────────────────────────────────────────────────────────────┘
"""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm, MAX_EXECUTION_ATTEMPTS
from agent.state import SingleFileState, create_single_file_state
from tools.agent_tools import (
    get_tools_for_agent,
    ANALYZER_TOOLS,
    FIXER_TOOLS,
    EXECUTOR_TOOLS
)


# ============================================================================
# SUPERVISOR PROMPT FOR SINGLE FILE
# ============================================================================

SINGLE_FILE_SUPERVISOR_PROMPT = """You are the Supervisor for a single-file code improvement agent.

Your role is to OBSERVE the current state and DECIDE what to do next.
You delegate to specialist agents - you do NOT execute tasks yourself.

## Available Agents

1. **analyzer** - Analyze code for syntax errors and issues
   - Use when: Need to check code quality, find errors
   - Tools: check_python_syntax, read_file_with_lines

2. **fixer** - Fix code issues with precise edits
   - Use when: Errors found that need fixing
   - Tools: str_replace, insert_at_line, delete_lines
   - IMPORTANT: Uses precise editing, NOT file overwrite!

3. **executor** - Run code to verify it works
   - Use when: Need to test if code executes correctly
   - Tools: execute_python_file

## Decision Rules

1. Start with **analyzer** to understand code issues
2. Use **fixer** when errors are identified
3. Use **executor** after fixing to verify
4. Loop back if execution fails (fixer -> executor -> ...)
5. **FINISH** when:
   - Code executes successfully
   - Maximum attempts reached
   - No more fixable issues

## Your Output Format

Respond with valid JSON:
```json
{
    "reasoning": "Your analysis of current state",
    "decision": "analyzer OR fixer OR executor OR FINISH",
    "task_for_agent": "Specific instructions",
    "confidence": "high/medium/low"
}
```

## Important
- Be decisive
- Don't loop indefinitely
- If same error repeats 3+ times, try different approach or FINISH
- Prefer precise edits (str_replace) over rewrites
"""


STATE_SUMMARY_TEMPLATE = """## Current State

### Target File
- Path: {target_file}
- Lines: {total_lines}

### Analysis Status
- Analyzed: {analysis_complete}
- Issues found: {issues_count}

### Execution Status
- Attempts: {execution_attempts}/{max_attempts}
- Last result: {last_result}
- Success: {execution_success}

### Last Error
{last_error}

### Modification History
{modification_count} modifications made

### Recent Activity
{recent_activity}

---

What should be the next action?
"""


# ============================================================================
# SINGLE FILE AGENT CLASS
# ============================================================================

class SingleFileAgent:
    """
    单文件代码分析的自主代理
    
    特点：
    1. Supervisor自主决策
    2. 专注于单文件分析和修复
    3. 使用精确编辑工具
    4. 支持反馈循环
    """
    
    AVAILABLE_AGENTS = ["analyzer", "fixer", "executor"]
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        max_attempts: int = MAX_EXECUTION_ATTEMPTS
    ):
        """
        初始化单文件代理
        
        Args:
            llm_provider: LLM提供商
            max_attempts: 最大尝试次数
        """
        self.llm_provider = llm_provider
        self.max_attempts = max_attempts
        
        # 创建LLM实例
        self.supervisor_llm = get_llm(llm_provider, "powerful")
        self.worker_llm = get_llm(llm_provider, "default")
        
        # 创建专家代理
        self.specialists = {
            "analyzer": self._create_agent_executor("analyzer", self.worker_llm),
            "fixer": self._create_agent_executor("fixer", self.supervisor_llm),
            "executor": self._create_agent_executor("executor", self.worker_llm),
        }
        
        # 内存检查点
        self.checkpointer = MemorySaver()
        
        # 构建图
        self.graph = self._build_graph()
    
    def _create_agent_executor(self, agent_name: str, llm):
        """创建代理执行器"""
        tools = get_tools_for_agent(agent_name)
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        
        def executor(state: SingleFileState, task: str) -> Dict[str, Any]:
            """执行代理任务"""
            
            # 构建提示
            if agent_name == "fixer":
                prompt = f"""You are the Fixer Agent.

Task: {task}

## CRITICAL: Use str_replace for edits!

1. First read the file with read_file_with_lines
2. Find the EXACT text to replace (including whitespace)
3. Use str_replace(file_path, old_str, new_str)

DO NOT rewrite the entire file!

Target file: {state['target_file']}
Last error: {state.get('last_error', 'None')}
"""
            elif agent_name == "analyzer":
                prompt = f"""You are the Analyzer Agent.

Task: {task}

Analyze the code for:
1. Syntax errors
2. Import errors
3. Type issues
4. Logic problems

Target file: {state['target_file']}
"""
            else:  # executor
                prompt = f"""You are the Executor Agent.

Task: {task}

Execute the code and report:
1. Success or failure
2. Output produced
3. Error details if any

Target file: {state['target_file']}
"""
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Execute the task for file: {state['target_file']}")
            ]
            
            # 工具调用循环
            results = []
            max_turns = 5
            
            for turn in range(max_turns):
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                content = response.content if hasattr(response, 'content') else str(response)
                
                print(f"    [{agent_name.upper()}] Turn {turn+1}: {content[:100]}...")
                
                if not hasattr(response, 'tool_calls') or not response.tool_calls:
                    results.append({"type": "response", "content": content})
                    break
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    result = f"Tool {tool_name} not found"
                    for t in tools:
                        if t.name == tool_name:
                            try:
                                result = t.invoke(tool_args)
                                print(f"      Tool [{tool_name}]: OK")
                            except Exception as e:
                                result = f"Error: {str(e)}"
                                print(f"      Tool [{tool_name}]: ERROR")
                            break
                    
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                    results.append({
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": str(result)[:1000]
                    })
            
            return {
                "agent": agent_name,
                "results": results,
                "messages": messages[1:]
            }
        
        return executor
    
    def _build_graph(self) -> StateGraph:
        """构建代理图"""
        workflow = StateGraph(SingleFileState)
        
        # 添加节点
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("fixer", self._fixer_node)
        workflow.add_node("executor", self._executor_node)
        
        # 入口点
        workflow.add_edge(START, "supervisor")
        
        # Supervisor动态路由
        workflow.add_conditional_edges(
            "supervisor",
            self._route_decision,
            {
                "analyzer": "analyzer",
                "fixer": "fixer",
                "executor": "executor",
                "FINISH": END
            }
        )
        
        # 所有代理返回Supervisor
        for agent in self.AVAILABLE_AGENTS:
            workflow.add_edge(agent, "supervisor")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _supervisor_node(self, state: SingleFileState) -> Dict[str, Any]:
        """Supervisor决策节点"""
        attempts = state.get("execution_attempts", 0)
        
        print(f"\n{'='*60}")
        print(f"  SUPERVISOR - Attempt {attempts + 1}/{self.max_attempts}")
        print(f"{'='*60}")
        
        # 检查是否完成
        if state.get("execution_success", False):
            print("  [+] Code executes successfully - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "final_status": f"SUCCESS after {attempts} attempt(s)"
            }
        
        if attempts >= self.max_attempts:
            print("  [!] Max attempts reached - FINISHING")
            return {
                "supervisor_decision": "FINISH",
                "final_status": f"INCOMPLETE: Max attempts ({self.max_attempts}) reached"
            }
        
        # 格式化状态摘要
        state_summary = self._format_state_summary(state)
        
        # 询问Supervisor
        messages = [
            SystemMessage(content=SINGLE_FILE_SUPERVISOR_PROMPT),
            HumanMessage(content=state_summary)
        ]
        
        response = self.supervisor_llm.invoke(messages)
        content = response.content
        
        # 解析决策
        decision = self._parse_decision(content)
        
        print(f"  [Decision] -> {decision['decision']}")
        
        return {
            "supervisor_decision": decision["decision"],
            "current_task": decision.get("task_for_agent", ""),
            "messages": [response]
        }
    
    def _format_state_summary(self, state: SingleFileState) -> str:
        """格式化状态摘要"""
        # 读取文件行数
        try:
            content = Path(state["target_file"]).read_text(encoding='utf-8')
            total_lines = len(content.splitlines())
        except:
            total_lines = 0
        
        recent = state.get("modification_history", [])[-3:]
        recent_activity = "\n".join(
            f"  - Modification {i+1}"
            for i, _ in enumerate(recent)
        ) if recent else "  (No activity)"
        
        return STATE_SUMMARY_TEMPLATE.format(
            target_file=state.get("target_file", "Unknown"),
            total_lines=total_lines,
            analysis_complete=state.get("analysis_complete", False),
            issues_count=len(state.get("identified_issues", [])),
            execution_attempts=state.get("execution_attempts", 0),
            max_attempts=self.max_attempts,
            last_result=state.get("last_execution_result", "None")[:200],
            execution_success=state.get("execution_success", False),
            last_error=state.get("last_error", "None")[:300],
            modification_count=len(state.get("modification_history", [])),
            recent_activity=recent_activity
        )
    
    def _parse_decision(self, content: str) -> Dict[str, Any]:
        """解析Supervisor决策"""
        try:
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        content_lower = content.lower()
        
        for agent in self.AVAILABLE_AGENTS:
            if agent in content_lower:
                return {"decision": agent, "task_for_agent": content[:200]}
        
        return {"decision": "FINISH", "task_for_agent": ""}
    
    def _route_decision(self, state: SingleFileState) -> str:
        """路由决策"""
        decision = state.get("supervisor_decision", "FINISH")
        if decision not in self.AVAILABLE_AGENTS + ["FINISH"]:
            return "FINISH"
        return decision
    
    def _analyzer_node(self, state: SingleFileState) -> Dict[str, Any]:
        """Analyzer节点"""
        task = state.get("current_task", "Analyze the code for errors")
        result = self.specialists["analyzer"](state, task)
        
        # 解析分析结果
        issues = []
        analysis_text = ""
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result":
                try:
                    data = json.loads(r.get("result", "{}"))
                    if not data.get("valid", True):
                        issues.append(data.get("error", "Unknown error"))
                except:
                    pass
            elif r.get("type") == "response":
                analysis_text = r.get("content", "")
        
        return {
            "code_analysis": analysis_text,
            "identified_issues": issues,
            "analysis_complete": True,
            "messages": result.get("messages", [])
        }
    
    def _fixer_node(self, state: SingleFileState) -> Dict[str, Any]:
        """Fixer节点"""
        task = state.get("current_task", "Fix the identified errors")
        result = self.specialists["fixer"](state, task)
        
        # 记录修改
        modification_history = list(state.get("modification_history", []))
        modification_history.append({
            "attempt": state.get("execution_attempts", 0) + 1,
            "task": task,
            "timestamp": datetime.now().isoformat()
        })
        
        # 检查是否使用了str_replace
        used_precise = any(
            r.get("tool") == "str_replace"
            for r in result.get("results", [])
            if r.get("type") == "tool_result"
        )
        
        return {
            "modification_history": modification_history,
            "messages": result.get("messages", [])
        }
    
    def _executor_node(self, state: SingleFileState) -> Dict[str, Any]:
        """Executor节点"""
        task = state.get("current_task", "Execute the code to verify")
        result = self.specialists["executor"](state, task)
        
        # 解析执行结果
        success = False
        error = ""
        output = ""
        
        for r in result.get("results", []):
            if r.get("type") == "tool_result":
                try:
                    data = json.loads(r.get("result", "{}"))
                    success = data.get("success", False)
                    output = data.get("stdout", "")
                    error = data.get("stderr", "") or data.get("error", "")
                except:
                    result_str = r.get("result", "")
                    success = "SUCCESS" in result_str
                    if not success:
                        error = result_str
        
        return {
            "execution_attempts": state.get("execution_attempts", 0) + 1,
            "execution_success": success,
            "last_execution_result": output[:500],
            "last_error": error[:500] if not success else "",
            "messages": result.get("messages", [])
        }
    
    def run(
        self,
        file_path: str,
        initial_code: str = "",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行单文件分析
        
        Args:
            file_path: 文件路径
            initial_code: 初始代码（可选）
            thread_id: 线程ID（可选）
            
        Returns:
            最终状态
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"  SINGLE FILE AGENT")
        print(f"{'='*80}")
        print(f"  Target File:   {file_path}")
        print(f"  Max Attempts:  {self.max_attempts}")
        print(f"  LLM Provider:  {self.llm_provider}")
        print(f"  Thread ID:     {thread_id}")
        print(f"{'='*80}")
        
        # 创建初始状态
        initial_state = create_single_file_state(
            target_file=file_path,
            initial_code=initial_code,
            max_attempts=self.max_attempts
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = self.graph.invoke(initial_state, config)
            
            print(f"\n{'='*80}")
            print(f"  AGENT COMPLETE")
            print(f"{'='*80}")
            print(f"  Attempts: {final_state.get('execution_attempts', 0)}")
            print(f"  Success: {final_state.get('execution_success', False)}")
            print(f"  Status: {final_state.get('final_status', 'Unknown')}")
            print(f"{'='*80}")
            
            return final_state
            
        except Exception as e:
            print(f"\n  [ERROR] {str(e)}")
            raise
    
    def stream_run(
        self,
        file_path: str,
        initial_code: str = "",
        thread_id: Optional[str] = None
    ):
        """流式运行"""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        initial_state = create_single_file_state(
            target_file=file_path,
            initial_code=initial_code,
            max_attempts=self.max_attempts
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_workflow(
    llm_provider: str = "openrouter",
    max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    **kwargs
) -> SingleFileAgent:
    """
    创建单文件代理的工厂函数
    
    Args:
        llm_provider: LLM提供商
        max_attempts: 最大尝试次数
        
    Returns:
        SingleFileAgent实例
    """
    return SingleFileAgent(llm_provider, max_attempts)


# 向后兼容
CodeImprovementWorkflow = SingleFileAgent


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SingleFileAgent",
    "CodeImprovementWorkflow",
    "create_workflow",
]
