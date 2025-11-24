"""
Configuration for multi-agent coding assistant
Supports multiple LLM providers
"""
import os
from typing import Optional, Literal

# Model configuration
DEFAULT_MODEL_PROVIDER: Literal["anthropic", "openai", "google", "ollama"] = "anthropic"
DEFAULT_MODEL_NAME = "claude-3-7-sonnet-latest"

# Model mappings for different providers
MODEL_MAPPINGS = {
    "anthropic": {
        "default": "claude-3-7-sonnet-latest",
        "fast": "claude-3-haiku-20240307",
        "powerful": "claude-3-opus-20240229"
    },
    "openai": {
        "default": "gpt-4-turbo-preview",
        "fast": "gpt-3.5-turbo",
        "powerful": "gpt-4"
    },
    "google": {
        "default": "gemini-1.5-pro",
        "fast": "gemini-1.5-flash",
        "powerful": "gemini-1.5-pro"
    },
    "ollama": {
        "default": "llama3.1",
        "fast": "llama3.1:8b",
        "powerful": "llama3.1:70b"
    }
}

# Execution configuration
MAX_EXECUTION_ATTEMPTS = 5
EXECUTION_TIMEOUT = 30  # seconds
WORKING_DIR = "workspace"

# MCP configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"


def get_model_config(provider: Optional[str] = None, model_type: str = "default") -> tuple:
    """Get model configuration based on provider and type"""
    provider = provider or DEFAULT_MODEL_PROVIDER
    
    if provider not in MODEL_MAPPINGS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    model_name = MODEL_MAPPINGS[provider].get(model_type, MODEL_MAPPINGS[provider]["default"])
    return provider, model_name


def get_llm(provider: Optional[str] = None, model_type: str = "default"):
    """Initialize LLM based on provider"""
    provider, model_name = get_model_config(provider, model_type)
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0)
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

