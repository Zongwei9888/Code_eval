"""
Prompt Templates Module
Centralized prompt management for multi-agent system
"""

# Single-file workflow prompts
from .system_prompts import (
    ANALYZER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    MODIFIER_SYSTEM_PROMPT,
    EXECUTOR_SUMMARY_PROMPT,
    format_analyzer_prompt,
    format_executor_prompt,
    format_modifier_prompt,
    format_workflow_start_prompt
)

# Repository workflow prompts
from .repo_prompts import (
    REPO_SCANNER_SYSTEM_PROMPT,
    REPO_ANALYZER_SYSTEM_PROMPT,
    REPO_FIXER_SYSTEM_PROMPT,
    REPO_EXECUTOR_SYSTEM_PROMPT,
    REPO_TESTER_SYSTEM_PROMPT,
    REPO_REPORTER_SYSTEM_PROMPT,
    format_scan_prompt,
    format_analyze_prompt,
    format_fix_prompt,
    format_execute_prompt,
    format_test_prompt
)

__all__ = [
    # Single-file prompts
    "ANALYZER_SYSTEM_PROMPT",
    "EXECUTOR_SYSTEM_PROMPT",
    "MODIFIER_SYSTEM_PROMPT",
    "EXECUTOR_SUMMARY_PROMPT",
    "format_analyzer_prompt",
    "format_executor_prompt",
    "format_modifier_prompt",
    "format_workflow_start_prompt",
    # Repository prompts
    "REPO_SCANNER_SYSTEM_PROMPT",
    "REPO_ANALYZER_SYSTEM_PROMPT",
    "REPO_FIXER_SYSTEM_PROMPT",
    "REPO_EXECUTOR_SYSTEM_PROMPT",
    "REPO_TESTER_SYSTEM_PROMPT",
    "REPO_REPORTER_SYSTEM_PROMPT",
    "format_scan_prompt",
    "format_analyze_prompt",
    "format_fix_prompt",
    "format_execute_prompt",
    "format_test_prompt"
]
