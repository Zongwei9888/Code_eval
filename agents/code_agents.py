"""
Multi-agent system for code analysis, execution, and modification
Each agent has a specific role in the code improvement workflow
"""
from typing import List, Dict, Any, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json

from .tools import ALL_TOOLS
from .config import get_llm


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


class MultiAgentState(MessagesState):
    """
    Shared state across all agents
    Extends MessagesState with custom fields for code workflow
    """
    # File information
    target_file: str = ""
    file_content: str = ""
    
    # Analysis results
    analysis_complete: bool = False
    code_analysis: str = ""
    identified_issues: List[str] = Field(default_factory=list)
    
    # Execution results
    execution_attempts: int = 0
    last_execution_result: str = ""
    execution_success: bool = False
    last_error: str = ""
    
    # Modification tracking
    modification_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_code: str = ""
    
    # Workflow control
    max_attempts: int = 5
    should_continue: bool = True
    final_status: str = ""


class CodeAnalyzerAgent:
    """Agent for analyzing code structure, quality, and potential issues"""
    
    def __init__(self, llm_provider: str = "anthropic"):
        self.llm = get_llm(llm_provider, "default")
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        self.system_prompt = """You are an expert code analyzer. Your role is to:
1. Analyze code structure and quality
2. Identify potential bugs, errors, and issues
3. Assess code complexity and maintainability
4. Suggest improvements

When analyzing code:
- Check for syntax errors
- Look for logical issues
- Identify missing imports or dependencies
- Assess code organization
- Consider edge cases and error handling

Provide a comprehensive analysis with specific, actionable findings."""

    def analyze(self, state: MultiAgentState) -> Dict[str, Any]:
        """Analyze code and identify issues"""
        
        # Read file if needed
        if not state.file_content and state.target_file:
            messages = state.messages + [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Read and analyze the file: {state.target_file}")
            ]
        else:
            messages = state.messages + [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""Analyze the following code:

File: {state.target_file}

Code:
```python
{state.file_content or state.current_code}
```

Provide detailed analysis including:
1. Code structure assessment
2. Potential issues or errors
3. Missing dependencies
4. Code quality observations
5. Specific recommendations for improvement""")
            ]
        
        response = self.llm_with_tools.invoke(messages)
        
        # Handle tool calls if any
        updated_messages = state.messages + [response]
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find and execute tool
                from .tools import get_tool_by_name
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
    
    def __init__(self, llm_provider: str = "anthropic"):
        self.llm = get_llm(llm_provider, "fast")
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        self.system_prompt = """You are a code execution specialist. Your role is to:
1. Execute code safely
2. Capture all output and errors
3. Analyze execution results
4. Provide clear error diagnosis

Use the available tools to execute code and analyze results.
Always provide detailed information about what succeeded and what failed."""

    def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute code and capture results"""
        
        code_to_execute = state.current_code or state.file_content
        file_path = state.target_file
        
        messages = state.messages + [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Execute the following code and analyze the results:

File: {file_path}

Code:
```python
{code_to_execute}
```

Use the execute_python_code tool to run this code.
Then analyze the execution result and provide a summary.""")
        ]
        
        response = self.llm_with_tools.invoke(messages)
        updated_messages = state.messages + [response]
        
        execution_result = ""
        success = False
        error_details = ""
        
        # Execute tools
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                from .tools import get_tool_by_name
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
            final_response = self.llm.invoke(updated_messages + [
                HumanMessage(content="Provide a brief summary of the execution results.")
            ])
            updated_messages.append(final_response)
        
        return {
            "messages": updated_messages,
            "execution_attempts": state.execution_attempts + 1,
            "last_execution_result": execution_result,
            "execution_success": success,
            "last_error": error_details if not success else ""
        }


class CodeModifierAgent:
    """Agent for modifying code to fix issues"""
    
    def __init__(self, llm_provider: str = "anthropic"):
        self.llm = get_llm(llm_provider, "powerful")
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        self.system_prompt = """You are an expert code modification specialist. Your role is to:
1. Fix bugs and errors in code
2. Improve code quality and structure
3. Add missing imports and dependencies
4. Ensure code follows best practices

When modifying code:
- Make minimal, targeted changes
- Preserve existing functionality
- Add comments explaining fixes
- Ensure the code will execute successfully
- Fix one issue at a time when possible

Always use the write_file_tool to save your modifications."""

    def modify(self, state: MultiAgentState) -> Dict[str, Any]:
        """Modify code to fix identified issues"""
        
        current_code = state.current_code or state.file_content
        analysis = state.code_analysis
        issues = state.identified_issues
        error = state.last_error
        
        context = f"""Current Code:
```python
{current_code}
```

Analysis: {analysis}

Identified Issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Last Execution Error:
{error}

Please fix the code to resolve these issues. Write the corrected code to the file: {state.target_file}"""
        
        messages = state.messages + [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=context)
        ]
        
        response = self.llm_with_tools.invoke(messages)
        updated_messages = state.messages + [response]
        
        modified_code = current_code
        changes_made = []
        
        # Handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                from .tools import get_tool_by_name
                tool = get_tool_by_name(tool_name)
                
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        
                        # If writing file, capture the new code
                        if tool_name == "write_file_tool" and "content" in tool_args:
                            modified_code = tool_args["content"]
                            changes_made.append(f"Modified {state.target_file}")
                        
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
            "attempt": state.execution_attempts,
            "issues_addressed": issues,
            "changes": changes_made,
            "code_snapshot": modified_code
        }
        
        return {
            "messages": updated_messages,
            "current_code": modified_code,
            "modification_history": state.modification_history + [modification_record]
        }


# Initialize agents (will be created when workflow is built)
def create_agents(llm_provider: str = "anthropic"):
    """Create all agent instances"""
    return {
        "analyzer": CodeAnalyzerAgent(llm_provider),
        "executor": CodeExecutorAgent(llm_provider),
        "modifier": CodeModifierAgent(llm_provider)
    }

