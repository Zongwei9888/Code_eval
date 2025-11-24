"""Prompt templates for multi-agent system"""
from .system_prompts import (
    ANALYZER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    MODIFIER_SYSTEM_PROMPT,
    format_analyzer_prompt,
    format_executor_prompt,
    format_modifier_prompt,
    format_workflow_start_prompt
)

__all__ = [
    "ANALYZER_SYSTEM_PROMPT",
    "EXECUTOR_SYSTEM_PROMPT",
    "MODIFIER_SYSTEM_PROMPT",
    "format_analyzer_prompt",
    "format_executor_prompt",
    "format_modifier_prompt",
    "format_workflow_start_prompt"
]

