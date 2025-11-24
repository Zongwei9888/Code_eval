# Multi-Agent Code Assistant 🤖

A sophisticated LangGraph-based system that automatically analyzes, executes, and fixes Python code until it runs successfully. Built with multiple specialized AI agents working together to improve your code.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for analysis, execution, and modification
- **Iterative Improvement**: Automatically fixes code through multiple attempts
- **Universal LLM Support**: Works with Anthropic, OpenAI, Google, and Ollama models
- **MCP Integration**: Ready for Model Context Protocol tool integration
- **Interactive Mode**: Engage with agents in real-time
- **Comprehensive Tooling**: File operations, code execution, error analysis
- **Workflow Visualization**: See the agent workflow graphically

## 🏗️ Architecture

The system consists of three main agents:

1. **Code Analyzer Agent** 🔍
   - Analyzes code structure and quality
   - Identifies potential bugs and issues
   - Assesses complexity and maintainability

2. **Code Executor Agent** ▶️
   - Safely executes code
   - Captures output and errors
   - Provides detailed diagnostics

3. **Code Modifier Agent** 🔧
   - Fixes identified issues
   - Improves code quality
   - Ensures best practices

### Workflow

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────────┐
│   Analyze   │ ◄───────┐
│    Code     │         │
└──────┬──────┘         │
       │                │
       ▼                │
┌─────────────┐         │
│   Execute   │         │
│    Code     │         │
└──────┬──────┘         │
       │                │
       ▼                │
   ┌───────┐            │
   │Success?│──No───────┤
   └───┬───┘            │
       │                │
      Yes               │
       │           ┌─────────┐
       │           │ Modify  │
       │           │  Code   │
       │           └────┬────┘
       │                │
       ▼                │
  ┌────────┐            │
  │  END   │            │
  └────────┘            │
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended) or pip

### Setup

1. **Create and activate conda environment:**

```bash
conda create -n langgraph python=3.11
conda activate langgraph
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure API keys:**

Create a `.env` file in the project root:

```bash
# For Anthropic (Claude)
ANTHROPIC_API_KEY=your_api_key_here

# For OpenAI
OPENAI_API_KEY=your_api_key_here

# For Google (Gemini)
GOOGLE_API_KEY=your_api_key_here

# MCP Configuration (optional)
MCP_ENABLED=true
MCP_SERVER_URL=http://localhost:8000
```

## 📖 Usage

### Command Line Interface

#### Fix a specific file:

```bash
python agents/main.py --file path/to/your/script.py
```

#### Interactive mode:

```bash
python agents/main.py --interactive
```

#### Use different LLM provider:

```bash
# Use OpenAI
python agents/main.py --file script.py --provider openai

# Use Google Gemini
python agents/main.py --file script.py --provider google

# Use Ollama (local)
python agents/main.py --file script.py --provider ollama
```

#### Customize behavior:

```bash
# Set max attempts
python agents/main.py --file script.py --max-attempts 10

# Use fast model
python agents/main.py --file script.py --model-type fast

# Stream updates
python agents/main.py --file script.py --stream

# Verbose output
python agents/main.py --file script.py --verbose
```

#### Visualize workflow:

```bash
python agents/main.py --visualize workflow_graph.png
```

### Programmatic Usage

```python
from agents import create_workflow

# Create workflow
workflow = create_workflow(llm_provider="anthropic", max_attempts=5)

# Run on a file
result = workflow.run(
    file_path="path/to/script.py",
    initial_code="print('Hello, World!')"
)

# Check results
if result["execution_success"]:
    print("✅ Code fixed successfully!")
else:
    print("⚠️ Could not fix code completely")
```

### Interactive Session

```python
from agents import interactive_session

# Start interactive mode
interactive_session(llm_provider="anthropic")
```

## 🛠️ Tools Available

The agents have access to the following tools:

- **File Operations**:
  - `read_file_tool`: Read file contents
  - `write_file_tool`: Write to files
  - `list_files_tool`: List directory contents

- **Code Execution**:
  - `execute_python_code`: Execute code snippets
  - `execute_file`: Execute Python files

- **Analysis**:
  - `analyze_error`: Analyze error messages
  - `search_code`: Search for terms in code

- **MCP Integration**:
  - `call_mcp_tool`: Call external MCP tools

## 🔧 Configuration

Edit `agents/config.py` to customize:

- Default model providers
- Model mappings
- Execution timeouts
- MCP settings

```python
# Example configuration
DEFAULT_MODEL_PROVIDER = "anthropic"
MAX_EXECUTION_ATTEMPTS = 5
EXECUTION_TIMEOUT = 30  # seconds
```

## 📊 Supported LLM Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| Anthropic | Claude 3.5 Sonnet, Opus, Haiku | Yes |
| OpenAI | GPT-4, GPT-3.5 Turbo | Yes |
| Google | Gemini 1.5 Pro, Flash | Yes |
| Ollama | Llama 3.1, Custom models | No (local) |

## 🎯 Example Use Cases

### 1. Fix Syntax Errors

```bash
python agents/main.py --file buggy_script.py
```

The system will:
- Detect syntax errors
- Suggest and apply fixes
- Verify the code runs

### 2. Add Missing Imports

```bash
python agents/main.py --file incomplete_code.py
```

The system will:
- Identify missing modules
- Add required imports
- Test execution

### 3. Debug Runtime Errors

```bash
python agents/main.py --file error_prone.py
```

The system will:
- Execute the code
- Capture runtime errors
- Apply fixes iteratively

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Test specific components:

```bash
# Test tools
pytest tests/test_tools.py

# Test agents
pytest tests/test_agents.py

# Test workflow
pytest tests/test_workflow.py
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) - LLM application framework
- [LangGraph](https://www.langchain.com/langgraph) - Agent workflow orchestration
- [Anthropic Claude](https://www.anthropic.com/) - Advanced AI models

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review example files in `examples/`

## 🔮 Future Enhancements

- [ ] Support for more programming languages
- [ ] Enhanced MCP tool integration
- [ ] Web UI for interactive sessions
- [ ] Code quality metrics
- [ ] Test generation
- [ ] Performance profiling
- [ ] Multi-file project support
- [ ] Git integration

---

**Made with ❤️ using LangGraph and Claude**

