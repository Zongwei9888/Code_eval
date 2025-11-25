"""
Multi-agent system for code analysis, execution, and modification
Each agent has a specific role in the code improvement workflow
"""
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from tools import ALL_TOOLS, get_tool_by_name
from tools.mcp_integration import get_mcp_tools_sync
from config import get_llm
from prompt import (
    ANALYZER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    MODIFIER_SYSTEM_PROMPT,
    format_analyzer_prompt,
    format_executor_prompt,
    format_modifier_prompt
)
from agent.state import MultiAgentState


class CodeAnalysisState(BaseModel):
    """State for code analysis"""
    file_path: str
    code_content: str
    analysis: str = ""
    issues: List[str] = Field(default_factory=list)
    complexity_score: int = 0


class CodeExecutionState(BaseModel):
    """State for code execution"""
    file_path: str
    code_content: str
    execution_result: str = ""
    success: bool = False
    error_details: str = ""
    attempt_number: int = 0


class CodeModificationState(BaseModel):
    """State for code modification"""
    file_path: str
    original_code: str
    modified_code: str = ""
    modification_reason: str = ""
    changes_made: List[str] = Field(default_factory=list)


class CodeAnalyzerAgent:
    """Agent for analyzing code structure, quality, and potential issues"""
    
    def __init__(self, llm_provider: str = "openrouter"):
        self.llm = get_llm(llm_provider, "default")
        
        # Combine standard tools with MCP tools
        all_tools = ALL_TOOLS.copy()
        try:
            mcp_tools = get_mcp_tools_sync()
            if mcp_tools:
                all_tools.extend(mcp_tools)
                print(f"[+] CodeAnalyzerAgent: Added {len(mcp_tools)} MCP tools")
        except Exception:
            pass  # MCP is optional
        
        self.llm_with_tools = self.llm.bind_tools(all_tools)
        self.system_prompt = ANALYZER_SYSTEM_PROMPT

    def analyze(self, state: MultiAgentState) -> Dict[str, Any]:
        """Analyze code and identify issues"""
        
        # Access state as dictionary
        file_content = state.get("file_content", "")
        target_file = state.get("target_file", "")
        current_code = state.get("current_code", "")
        messages = state.get("messages", [])
        
        # Read file if needed
        if not file_content and target_file:
            messages_to_send = list(messages) + [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Read and analyze the file: {target_file}")
            ]
        else:
            analysis_prompt = format_analyzer_prompt(
                file_path=target_file,
                code_content=file_content or current_code
            )
            messages_to_send = list(messages) + [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
        
        response = self.llm_with_tools.invoke(messages_to_send)
        
        # Handle tool calls if any
        updated_messages = list(messages) + [response]
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find and execute tool
                tool = get_tool_by_name(tool_name)
                
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        updated_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        updated_messages.append(
                            ToolMessage(
                                content=f"Error executing tool: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
            
            # Get final analysis after tool execution
            final_response = self.llm.invoke(updated_messages)
            updated_messages.append(final_response)
        
        # Extract analysis from response
        analysis_text = response.content if isinstance(response.content, str) else str(response.content)
        
        # Parse issues from analysis
        issues = self._extract_issues(analysis_text)
        
        return {
            "messages": updated_messages,
            "code_analysis": analysis_text,
            "identified_issues": issues,
            "analysis_complete": True
        }
    
    def _extract_issues(self, analysis: str) -> List[str]:
        """Extract specific issues from analysis text"""
        issues = []
        keywords = ["error", "issue", "problem", "bug", "missing", "incorrect"]
        
        lines = analysis.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                issues.append(line.strip())
        
        return issues[:10]  # Limit to top 10 issues


class CodeExecutorAgent:
    """Agent for executing code and capturing results/errors"""
    
    def __init__(self, llm_provider: str = "openrouter"):
        self.llm = get_llm(llm_provider, "fast")
        
        # Combine standard tools with MCP tools
        all_tools = ALL_TOOLS.copy()
        try:
            mcp_tools = get_mcp_tools_sync()
            if mcp_tools:
                all_tools.extend(mcp_tools)
                print(f"[+] CodeExecutorAgent: Added {len(mcp_tools)} MCP tools")
        except Exception:
            pass  # MCP is optional
        
        self.llm_with_tools = self.llm.bind_tools(all_tools)
        self.system_prompt = EXECUTOR_SYSTEM_PROMPT

    def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute code and capture results"""
        
        # Access state as dictionary
        current_code = state.get("current_code", "")
        file_content = state.get("file_content", "")
        target_file = state.get("target_file", "")
        messages = state.get("messages", [])
        execution_attempts = state.get("execution_attempts", 0)
        
        code_to_execute = current_code or file_content
        
        execution_prompt = format_executor_prompt(
            file_path=target_file,
            code_content=code_to_execute
        )
        
        messages_to_send = list(messages) + [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=execution_prompt)
        ]
        
        response = self.llm_with_tools.invoke(messages_to_send)
        updated_messages = list(messages) + [response]
        
        execution_result = ""
        success = False
        error_details = ""
        
        # Execute tools
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                tool = get_tool_by_name(tool_name)
                
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        execution_result = str(result)
                        
                        # Check if execution was successful
                        if "SUCCESS" in result:
                            success = True
                        elif "FAILED" in result or "ERROR" in result:
                            success = False
                            error_details = result
                        
                        updated_messages.append(
                            ToolMessage(
                                content=execution_result,
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        error_details = f"Tool execution error: {str(e)}"
                        updated_messages.append(
                            ToolMessage(
                                content=error_details,
                                tool_call_id=tool_call["id"]
                            )
                        )
            
            # Get final summary
            from prompt.system_prompts import EXECUTOR_SUMMARY_PROMPT
            final_response = self.llm.invoke(updated_messages + [
                HumanMessage(content=EXECUTOR_SUMMARY_PROMPT)
            ])
            updated_messages.append(final_response)
        
        return {
            "messages": updated_messages,
            "execution_attempts": execution_attempts + 1,
            "last_execution_result": execution_result,
            "execution_success": success,
            "last_error": error_details if not success else ""
        }


class CodeModifierAgent:
    """Agent for modifying code to fix issues"""
    
    def __init__(self, llm_provider: str = "openrouter"):
        self.llm = get_llm(llm_provider, "powerful")
        
        # Combine standard tools with MCP tools
        all_tools = ALL_TOOLS.copy()
        try:
            mcp_tools = get_mcp_tools_sync()
            if mcp_tools:
                all_tools.extend(mcp_tools)
                print(f"[+] CodeModifierAgent: Added {len(mcp_tools)} MCP tools")
        except Exception:
            pass  # MCP is optional
        
        self.llm_with_tools = self.llm.bind_tools(all_tools)
        self.system_prompt = MODIFIER_SYSTEM_PROMPT

    def modify(self, state: MultiAgentState) -> Dict[str, Any]:
        """Modify code to fix identified issues"""
        
        # Access state as dictionary
        current_code = state.get("current_code", "")
        file_content = state.get("file_content", "")
        target_file = state.get("target_file", "")
        code_analysis = state.get("code_analysis", "")
        identified_issues = state.get("identified_issues", [])
        last_error = state.get("last_error", "")
        messages = state.get("messages", [])
        execution_attempts = state.get("execution_attempts", 0)
        modification_history = state.get("modification_history", [])
        
        code = current_code or file_content
        
        modification_prompt = format_modifier_prompt(
            file_path=target_file,
            current_code=code,
            analysis=code_analysis,
            issues=identified_issues,
            error=last_error
        )
        
        messages_to_send = list(messages) + [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=modification_prompt)
        ]
        
        response = self.llm_with_tools.invoke(messages_to_send)
        updated_messages = list(messages) + [response]
        
        modified_code = code
        changes_made = []
        
        # Handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                tool = get_tool_by_name(tool_name)
                
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        
                        # If writing file, capture the new code
                        if tool_name == "write_file_tool" and "content" in tool_args:
                            modified_code = tool_args["content"]
                            changes_made.append(f"Modified {target_file}")
                        
                        updated_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        updated_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
        
        # Track modification history
        modification_record = {
            "attempt": execution_attempts,
            "issues_addressed": identified_issues,
            "changes": changes_made,
            "code_snapshot": modified_code
        }
        
        return {
            "messages": updated_messages,
            "current_code": modified_code,
            "modification_history": modification_history + [modification_record]
        }


# Initialize agents (will be created when workflow is built)
def create_agents(llm_provider: str = "openrouter"):
    """
    Create all agent instances
    
    Args:
        llm_provider: LLM provider to use
        
    Returns:
        Dictionary of agent instances
    """
    return {
        "analyzer": CodeAnalyzerAgent(llm_provider),
        "executor": CodeExecutorAgent(llm_provider),
        "modifier": CodeModifierAgent(llm_provider)
    }
