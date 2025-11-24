"""Configuration module for multi-agent system"""
from .llm_config import (
    get_llm,
    get_model_config,
    validate_config,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    MAX_EXECUTION_ATTEMPTS,
    EXECUTION_TIMEOUT,
    MCP_ENABLED,
    MCP_SERVER_URL,
    MODEL_MAPPINGS
)

__all__ = [
    "get_llm",
    "get_model_config",
    "validate_config",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL",
    "MAX_EXECUTION_ATTEMPTS",
    "EXECUTION_TIMEOUT",
    "MCP_ENABLED",
    "MCP_SERVER_URL",
    "MODEL_MAPPINGS"
]

