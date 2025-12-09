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

Your role: OBSERVE agent feedback → DECIDE next action → DELEGATE to agent.

## Available Agents (3 Core)

1. **analyzer** - Analyze code for errors and issues
   - Tools: check_python_syntax, read_file_with_lines

2. **fixer** - Fix code with precise edits (str_replace)
   - Tools: str_replace, insert_at_line, delete_lines

3. **executor** - Execute code to verify functionality
   - Tools: execute_python_file

## Decision Flow (Dynamic!)

Standard: analyzer → fixer (if errors) → executor → FINISH
          ↑                                   ↓
          └─────── (if failed) ───────────────┘

## Key Rules

1. **READ agent feedback first!** - Last agent tells you what to do next
2. **No hard-coded sequence** - Decide based on agent output
3. **Trust agent recommendations** - They know their domain
4. **NEVER loop same agent 3+ times** - Try different approach
5. **FINISH when**:
   - Executor succeeded
   - Max attempts reached
   - Same error repeats 3+ times

## Output Format

```json
{
    "reasoning": "Based on [last_agent] feedback: [summary]",
    "decision": "analyzer|fixer|executor|FINISH",
    "task_for_agent": "Specific task instructions"
}
```

## Decision Examples

**Agent: analyzer, Feedback: "Found 2 syntax errors"**
→ Decision: "fixer" (fix the errors)

**Agent: fixer, Feedback: "Fix applied, recommend executor"**
→ Decision: "executor" (verify the fix)

**Agent: executor, Feedback: "Failed with NameError"**
→ Decision: "fixer" (add missing definition)

**Agent: executor, Feedback: "Execution successful"**
→ Decision: "FINISH" (goal achieved)

**CRITICAL**: Always base decision on agent feedback, not hard rules!
"""


STATE_SUMMARY_TEMPLATE = """## Current State

### Last Agent Communication (READ THIS FIRST!)
{agent_communication}

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

---

Based on the agent feedback above, what should be the next action?
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
        """
        Supervisor决策节点 - 动态交互版本
        
        观察 agent 反馈并做出智能决策
        """
        attempts = state.get("execution_attempts", 0)
        last_agent = state.get("last_agent")
        agent_feedback = state.get("agent_feedback", "")
        
        print(f"\n{'='*60}")
        print(f"  SUPERVISOR - Attempt {attempts + 1}/{self.max_attempts}")
        if last_agent:
            print(f"  Last agent: {last_agent.upper()}")
            print(f"  Feedback: {agent_feedback[:100]}...")
        print(f"{'='*60}")
        
        # 检查循环（同一个 agent 连续 3 次）
        decision_history = state.get("decision_history", [])
        if len(decision_history) >= 3:
            last_3 = [d["decision"] for d in decision_history[-3:]]
            if len(set(last_3)) == 1:
                agent = last_3[0]
                print(f"  [!] LOOP DETECTED: {agent} called 3 times!")
                
                # 打破循环
                if agent == "analyzer":
                    print(f"  [!] Breaking loop: analyzer -> executor")
                    return {
                        "supervisor_decision": "executor",
                        "current_task": "Execute the code despite analysis loop",
                        "final_status": ""
                    }
                elif agent == "fixer":
                    print(f"  [!] Breaking loop: fixer -> executor (test the fix)")
                    return {
                        "supervisor_decision": "executor",
                        "current_task": "Verify if repeated fixes resolved the issue",
                        "final_status": ""
                    }
                elif agent == "executor":
                    print(f"  [!] Breaking loop: executor -> FINISH (stuck)")
                    return {
                        "supervisor_decision": "FINISH",
                        "final_status": f"INCOMPLETE: Stuck in executor loop after {attempts} attempts"
                    }
        
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
        
        # 格式化状态摘要（包含 agent 反馈）
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
        print(f"  [Reasoning] {decision.get('reasoning', 'N/A')[:100]}...")
        
        # 记录决策
        decision_history = list(decision_history)
        decision_history.append({
            "iteration": len(decision_history) + 1,
            "decision": decision["decision"],
            "reasoning": decision.get("reasoning", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "supervisor_decision": decision["decision"],
            "current_task": decision.get("task_for_agent", ""),
            "decision_history": decision_history,
            "messages": [response]
        }
    
    def _format_state_summary(self, state: SingleFileState) -> str:
        """
        格式化状态摘要 - 包含 agent 通信
        """
        # Agent communication
        last_agent = state.get("last_agent")
        agent_feedback = state.get("agent_feedback", "")
        last_agent_output = state.get("last_agent_output", {})
        context_for_next = state.get("context_for_next_agent", {})
        
        if last_agent:
            agent_communication = f"""
**Last Agent**: {last_agent.upper()}
**Feedback**: {agent_feedback}

**Structured Output**:
{json.dumps(last_agent_output, indent=2)}
"""
            if context_for_next:
                agent_communication += f"""
**Context for Next Agent**:
{json.dumps(context_for_next, indent=2)}
"""
        else:
            agent_communication = "**No previous agent** - Starting fresh. Recommend analyzer first."
        
        # 读取文件行数
        try:
            content = Path(state["target_file"]).read_text(encoding='utf-8')
            total_lines = len(content.splitlines())
        except:
            total_lines = 0
        
        return STATE_SUMMARY_TEMPLATE.format(
            agent_communication=agent_communication,
            target_file=state.get("target_file", "Unknown"),
            total_lines=total_lines,
            analysis_complete=state.get("analysis_complete", False),
            issues_count=len(state.get("identified_issues", [])),
            execution_attempts=state.get("execution_attempts", 0),
            max_attempts=self.max_attempts,
            last_result=state.get("last_execution_result", "None")[:200],
            execution_success=state.get("execution_success", False),
            last_error=state.get("last_error", "None")[:300],
            modification_count=len(state.get("modification_history", []))
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
        """
        Analyzer节点 - 动态交互版本
        
        分析代码并生成反馈给 Supervisor
        """
        task = state.get("current_task", "Analyze the code for errors")
        context = state.get("context_for_next_agent", {})
        
        print(f"\n  [ANALYZER] Task: {task[:100]}...")
        if context:
            print(f"  [ANALYZER] Context: {list(context.keys())}")
        
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
        
        # 生成反馈
        if issues:
            feedback = f"Analysis complete: Found {len(issues)} issue(s). "
            feedback += f"First issue: {issues[0][:100]}. "
            feedback += "Recommendation: Call fixer to resolve errors."
        else:
            feedback = "Analysis complete: No issues detected. Code appears correct. "
            feedback += "Recommendation: Call executor to verify functionality."
        
        # 结构化输出
        structured_output = {
            "agent": "analyzer",
            "issues_found": len(issues),
            "has_issues": len(issues) > 0,
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # 准备上下文给下一个 agent
        next_context = {}
        if issues:
            next_context["issues_to_fix"] = issues
            next_context["priority_issue"] = issues[0]
            next_context["target_file"] = state.get("target_file")
        
        print(f"  [ANALYZER] Feedback: {feedback[:150]}...")
        
        return {
            "code_analysis": analysis_text,
            "identified_issues": issues,
            "analysis_complete": True,
            "messages": result.get("messages", []),
            # Agent communication
            "last_agent": "analyzer",
            "agent_feedback": feedback,
            "last_agent_output": structured_output,
            "context_for_next_agent": next_context
        }
    
    def _fixer_node(self, state: SingleFileState) -> Dict[str, Any]:
        """
        Fixer节点 - 动态交互版本
        
        修复代码并生成反馈
        """
        task = state.get("current_task", "Fix the identified errors")
        context = state.get("context_for_next_agent", {})
        
        print(f"\n  [FIXER] Task: {task[:100]}...")
        if context:
            print(f"  [FIXER] Context: {list(context.keys())}")
            if "priority_issue" in context:
                print(f"  [FIXER] Priority issue: {context['priority_issue'][:100]}")
        
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
        
        # 生成反馈
        if used_precise:
            feedback = "Fix applied using precise edit (str_replace). "
            feedback += "Code modified successfully. "
            feedback += "Recommendation: Call executor to verify the fix worked."
        else:
            feedback = "Fixer completed but no str_replace detected. "
            feedback += "May need to retry. "
            feedback += "Recommendation: Call executor to check current state."
        
        # 结构化输出
        structured_output = {
            "agent": "fixer",
            "used_precise_edit": used_precise,
            "modifications_count": len(modification_history),
            "timestamp": datetime.now().isoformat()
        }
        
        # 准备上下文
        next_context = {
            "modified_file": state.get("target_file"),
            "should_verify": True,
            "modification_count": len(modification_history)
        }
        
        print(f"  [FIXER] Feedback: {feedback[:150]}...")
        
        return {
            "modification_history": modification_history,
            "messages": result.get("messages", []),
            # Agent communication
            "last_agent": "fixer",
            "agent_feedback": feedback,
            "last_agent_output": structured_output,
            "context_for_next_agent": next_context
        }
    
    def _executor_node(self, state: SingleFileState) -> Dict[str, Any]:
        """
        Executor节点 - 动态交互版本
        
        执行代码并生成详细反馈
        """
        task = state.get("current_task", "Execute the code to verify")
        context = state.get("context_for_next_agent", {})
        
        print(f"\n  [EXECUTOR] Task: {task[:100]}...")
        if context:
            print(f"  [EXECUTOR] Context: {list(context.keys())}")
        
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
        
        # 生成反馈
        if success:
            feedback = "Execution successful! "
            feedback += f"Output: {output[:100]}... " if output else "No output produced. "
            feedback += "All checks passed. Code is working correctly. "
            feedback += "Recommendation: FINISH (goal achieved)."
        else:
            feedback = f"Execution failed: {error[:150]}... "
            # 分析错误类型
            error_lower = error.lower()
            if "syntaxerror" in error_lower or "indentationerror" in error_lower:
                feedback += "Error type: Syntax error. "
                feedback += "Recommendation: Call fixer to correct syntax."
            elif "nameerror" in error_lower or "attributeerror" in error_lower:
                feedback += "Error type: Name/Attribute error. "
                feedback += "Recommendation: Call fixer to add missing definitions."
            elif "modulenotfounderror" in error_lower or "importerror" in error_lower:
                feedback += "Error type: Import error. "
                feedback += "Recommendation: Call fixer to fix imports."
            else:
                feedback += "Error type: Runtime error. "
                feedback += "Recommendation: Call fixer to debug logic."
        
        # 结构化输出
        structured_output = {
            "agent": "executor",
            "execution_success": success,
            "has_output": bool(output),
            "has_error": bool(error),
            "attempts": state.get("execution_attempts", 0) + 1,
            "timestamp": datetime.now().isoformat()
        }
        
        # 准备上下文
        next_context = {}
        if not success:
            next_context["execution_error"] = error[:300]
            next_context["target_file"] = state.get("target_file")
            next_context["needs_fix"] = True
        
        print(f"  [EXECUTOR] Feedback: {feedback[:150]}...")
        
        return {
            "execution_attempts": state.get("execution_attempts", 0) + 1,
            "execution_success": success,
            "last_execution_result": output[:500],
            "last_error": error[:500] if not success else "",
            "messages": result.get("messages", []),
            # Agent communication
            "last_agent": "executor",
            "agent_feedback": feedback,
            "last_agent_output": structured_output,
            "context_for_next_agent": next_context
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
