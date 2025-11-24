# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The project is pre-configured with OpenRouter API. The API key is already set in `config/llm_config.py`.

## Basic Usage

### 1. Validate Configuration
```bash
python main.py --validate-config
```

### 2. Interactive Mode
```bash
python main.py --interactive
```

### 3. Fix a File
```bash
python main.py --file your_script.py
```

### 4. Run Examples
```bash
python example_usage.py
```

## Project Structure

```
Code_Eval/
├── agent/              # Agent implementations
├── workflow/           # Workflow orchestration
├── tools/              # Tool definitions
├── prompt/             # System prompts
├── config/             # Configuration (API keys here)
├── main.py             # CLI entry point
└── example_usage.py    # Usage examples
```

## Configuration

The system is pre-configured with OpenRouter. To change providers or settings, edit:
- `config/llm_config.py` - LLM configuration
- Environment variables in `.env` file (optional)

## Available Commands

```bash
# Interactive session
python main.py --interactive

# Fix a specific file
python main.py --file script.py

# Use different provider
python main.py --file script.py --provider anthropic

# Set max attempts
python main.py --file script.py --max-attempts 10

# Verbose output
python main.py --file script.py --verbose

# Stream updates
python main.py --file script.py --stream

# Visualize workflow
python main.py --visualize workflow.png
```

## Programmatic Usage

```python
from main import quick_fix

# Quick fix a file
result = quick_fix("script.py", provider="openrouter", max_attempts=5)

if result["execution_success"]:
    print("✅ Code fixed!")
else:
    print("⚠️ Needs manual review")
```

## Supported Providers

- **openrouter** (default) - Pre-configured
- **anthropic** - Requires ANTHROPIC_API_KEY
- **openai** - Requires OPENAI_API_KEY
- **google** - Requires GOOGLE_API_KEY
- **ollama** - Local, no API key needed

## Need Help?

- Check `example_usage.py` for more examples
- Read `README.md` for detailed documentation
- Run `python main.py --help` for CLI options

