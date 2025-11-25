"""
LLM Configuration for Multi-Agent Code Assistant
Supports multiple providers including OpenRouter
Updated with better MCP configuration support
"""
import os
from typing import Optional, Literal, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default provider configuration
DEFAULT_PROVIDER: Literal["openrouter", "anthropic", "openai", "google", "ollama"] = "openrouter"
DEFAULT_MODEL = "openai/gpt-4o"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model mappings for different providers
MODEL_MAPPINGS = {
    "openrouter": {
        "default": "anthropic/claude-sonnet-4",
        "fast": "anthropic/claude-sonnet-4",
        "powerful": "anthropic/claude-opus-4.5"
    },
    "anthropic": {
        "default": "claude-3-7-sonnet-latest",
        "fast": "claude-3-haiku-20240307",
        "powerful": "claude-3-opus-20240229"
    },
    "openai": {
        "default": "openai/gpt-4-turbo-preview",
        "fast": "openai/gpt-3.5-turbo",
        "powerful": "openai/gpt-4o"
    },
    "google": {
        "default": "google/gemini-2.5-pro",
        "fast": "google/gemini-2.5-flash",
        "powerful": "google/gemini-2.5-pro"
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

# Memory/Checkpointing configuration
USE_SQLITE_CHECKPOINTER = os.getenv("USE_SQLITE_CHECKPOINTER", "false").lower() == "true"
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.db")

# MCP configuration
MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
MCP_FILESYSTEM_PATH = os.getenv("MCP_FILESYSTEM_PATH", "")


def get_model_config(provider: Optional[str] = None, model_type: str = "default") -> tuple:
    """
    Get model configuration based on provider and type
    
    Args:
        provider: LLM provider name
        model_type: Model type (default, fast, or powerful)
        
    Returns:
        Tuple of (provider, model_name)
    """
    provider = provider or DEFAULT_PROVIDER
    
    if provider not in MODEL_MAPPINGS:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(MODEL_MAPPINGS.keys())}")
    
    model_name = MODEL_MAPPINGS[provider].get(model_type, MODEL_MAPPINGS[provider]["default"])
    return provider, model_name


def get_llm(provider: Optional[str] = None, model_type: str = "default"):
    """
    Initialize LLM based on provider
    
    Args:
        provider: LLM provider name
        model_type: Model type (default, fast, or powerful)
        
    Returns:
        Initialized LLM instance
    """
    provider, model_name = get_model_config(provider, model_type)
    
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ChatAnthropic(model=model_name, temperature=0)
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model=model_name, temperature=0)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_mcp_config() -> Dict[str, Any]:
    """
    Get MCP configuration from environment
    
    Returns:
        Dictionary with MCP configuration
    """
    return {
        "enabled": MCP_ENABLED,
        "server_url": MCP_SERVER_URL,
        "filesystem_path": MCP_FILESYSTEM_PATH
    }


def get_checkpointer_config() -> Dict[str, Any]:
    """
    Get checkpointer configuration
    
    Returns:
        Dictionary with checkpointer configuration
    """
    return {
        "use_sqlite": USE_SQLITE_CHECKPOINTER,
        "sqlite_path": SQLITE_DB_PATH
    }


def validate_config():
    """Validate configuration and print status"""
    print("=" * 60)
    print("LLM Configuration Status")
    print("=" * 60)
    print(f"Default Provider: {DEFAULT_PROVIDER}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"OpenRouter API Key: {'Set' if OPENROUTER_API_KEY else 'Not set'}")
    print(f"OpenRouter Base URL: {OPENROUTER_BASE_URL}")
    print(f"Max Attempts: {MAX_EXECUTION_ATTEMPTS}")
    print(f"Execution Timeout: {EXECUTION_TIMEOUT}s")
    print()
    print("Memory/Checkpointing:")
    print(f"  SQLite Checkpointer: {'Enabled' if USE_SQLITE_CHECKPOINTER else 'Disabled (using in-memory)'}")
    if USE_SQLITE_CHECKPOINTER:
        print(f"  SQLite DB Path: {SQLITE_DB_PATH}")
    print()
    print("MCP Configuration:")
    print(f"  MCP Enabled: {MCP_ENABLED}")
    if MCP_ENABLED:
        print(f"  MCP Server URL: {MCP_SERVER_URL}")
        if MCP_FILESYSTEM_PATH:
            print(f"  MCP Filesystem Path: {MCP_FILESYSTEM_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    validate_config()
