"""
Workflow Module
Autonomous Agent Systems for code analysis and improvement

This module contains:
1. SingleFileAgent - Single file analysis with autonomous decision
2. AutonomousRepoAgent - Repository analysis with 10 specialist agents

CORE ARCHITECTURE:
- Both agents use Supervisor LLM for autonomous decision-making
- No hardcoded workflows - LLM decides each step dynamically
- Each specialist agent has dedicated tools

Available Agents (AutonomousRepoAgent):
1. planner - Task decomposition
2. researcher - Code search and understanding
3. scanner - Project structure discovery
4. analyzer - Error detection
5. fixer - Precise code editing (str_replace)
6. executor - Code execution
7. tester - Test running
8. reviewer - Code quality review
9. environment - Dependency management
10. git - Version control

Backward compatibility:
- MultiAgentRepoWorkflow = AutonomousRepoAgent
- TrueAgent = AutonomousRepoAgent
- CodeImprovementWorkflow = SingleFileAgent
"""

# Single-file autonomous agent
from .code_workflow import (
    SingleFileAgent,
    CodeImprovementWorkflow,  # Alias
    create_workflow
)

# Repository autonomous agent system
from .repo_workflow import (
    AutonomousRepoAgent,
    MultiAgentRepoWorkflow,  # Alias
    TrueAgent,  # Alias
    create_multi_agent_workflow,
    create_true_agent,  # Alias
    create_specialist_executor,
    create_agent_executor  # Alias
)

__all__ = [
    # Single-file agent
    "SingleFileAgent",
    "CodeImprovementWorkflow",
    "create_workflow",
    # Repository agent
    "AutonomousRepoAgent",
    "MultiAgentRepoWorkflow",
    "TrueAgent",
    "create_multi_agent_workflow",
    "create_true_agent",
    "create_specialist_executor",
    "create_agent_executor",
]
