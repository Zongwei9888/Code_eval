# Multi-Agent Code Assistant 🤖

A sophisticated LangGraph-based system that automatically analyzes, executes, and fixes Python code until it runs successfully. Built with multiple specialized AI agents working together to improve your code.

## 🌟 Features

### Core Architecture
- **Multi-Agent System**: Specialized agents (Analyzer, Executor, Modifier) built on LangGraph
- **Memory/Persistence**: Checkpoint-based state management with SQLite or in-memory storage
- **Thread Management**: Multiple concurrent sessions with thread ID isolation
- **Streaming Support**: Real-time workflow updates and progress monitoring

### LLM & Tools
- **Universal LLM Support**: OpenRouter (default), Anthropic, OpenAI, Google, Ollama
- **MCP Integration**: Full Model Context Protocol support with langchain-mcp-adapters
- **Rich Toolset**: File operations, code execution, error analysis, MCP tools

### Advanced Features
- **Iterative Improvement**: Automatically fixes code through multiple attempts
- **Resume Capability**: Continue from checkpoints after interruptions
- **Workflow Visualization**: See the agent workflow graphically
- **Subgraph Support**: Handle complex multi-file scenarios (example included)
- **Interactive Mode**: Engage with agents in real-time

### Compliance
- ✅ Follows [LangGraph Official Patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- ✅ Implements [Memory Best Practices](https://docs.langchain.com/oss/python/langgraph/add-memory)
- ✅ Supports [MCP Tool Integration](https://docs.langchain.com/oss/python/langchain/mcp)
- ✅ Includes [Subgraph Examples](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)

## 🏗️ Project Structure

```
Code_Eval/
├── agent/                          # Agent implementations
│   ├── code_agents.py              # Analyzer, Executor, Modifier agents
│   ├── state.py                    # TypedDict state definition (LangGraph)
│   └── __init__.py
├── workflow/                       # Workflow orchestration
│   ├── code_workflow.py            # Basic workflow
│   ├── code_workflow_improved.py   # With memory/checkpointing
│   ├── subgraph_example.py         # Multi-file subgraph pattern
│   └── __init__.py
├── tools/                          # Tool definitions
│   ├── code_tools.py               # File ops, execution, analysis
│   ├── mcp_integration.py          # MCP tool manager
│   └── __init__.py
├── prompt/                         # System prompts
│   ├── system_prompts.py           # Centralized prompts
│   └── __init__.py
├── config/                         # Configuration
│   ├── llm_config.py               # LLM & MCP configs
│   └── __init__.py
├── main.py                         # CLI entry point
├── example_usage.py                # Usage examples
├── ARCHITECTURE.md                 # Detailed architecture docs
├── QUICK_START.md                  # Quick start guide
└── requirements.txt                # Dependencies
```

## 🤖 Agents

The system consists of three specialized agents:

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

Create a `.env` file in the project root (or use environment variables):

```bash
# OpenRouter Configuration (Default - Pre-configured)
OPENROUTER_API_KEY=sk-or-v1-...  # Already set in code
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Other LLM Providers (Optional)
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here

# Memory/Checkpointing
USE_SQLITE_CHECKPOINTER=false  # true for persistent storage
SQLITE_DB_PATH=checkpoints.db

# MCP Configuration (Optional)
MCP_ENABLED=false  # true to enable MCP tools
MCP_SERVER_URL=http://localhost:8000/mcp
MCP_FILESYSTEM_PATH=/tmp  # For filesystem MCP server
```

**Note**: OpenRouter is configured as the default provider and includes the API key in `config/llm_config.py`.

## 📖 Usage

### Command Line Interface

#### Validate configuration:

```bash
python main.py --validate-config
```

#### Fix a specific file:

```bash
python main.py --file path/to/your/script.py
```

#### Interactive mode:

```bash
python main.py --interactive
```

#### Use different LLM provider:

```bash
# Use OpenRouter (default)
python main.py --file script.py --provider openrouter

# Use OpenAI
python main.py --file script.py --provider openai

# Use Anthropic
python main.py --file script.py --provider anthropic

# Use Google Gemini
python main.py --file script.py --provider google

# Use Ollama (local)
python main.py --file script.py --provider ollama
```

#### Customize behavior:

```bash
# Set max attempts
python main.py --file script.py --max-attempts 10

# Use fast model
python main.py --file script.py --model-type fast

# Stream updates
python main.py --file script.py --stream

# Verbose output
python main.py --file script.py --verbose
```

#### Visualize workflow:

```bash
python main.py --visualize workflow_graph.png
```

### Programmatic Usage

```python
from workflow import create_workflow
from main import quick_fix

# Method 1: Quick fix
result = quick_fix("path/to/script.py", provider="openrouter", max_attempts=5)

# Method 2: Create workflow with memory
workflow = create_workflow(
    llm_provider="openrouter",
    max_attempts=5,
    use_sqlite=True,  # Enable persistent checkpointing
    sqlite_path="checkpoints.db"
)

# Run on a file with thread ID
result = workflow.run(
    file_path="path/to/script.py",
    initial_code="print('Hello, World!')",
    thread_id="session_001"  # For session management
)

# Check results
if result["execution_success"]:
    print("✅ Code fixed successfully!")
else:
    print("⚠️ Could not fix code completely")

# Resume from checkpoint (if interrupted)
resumed_result = workflow.resume(thread_id="session_001")

# Stream updates in real-time
for update in workflow.stream_run("script.py", thread_id="session_002"):
    print(f"State update: {update}")
```

### Interactive Session

```python
from workflow import interactive_session

# Start interactive mode
interactive_session(llm_provider="openrouter")
```

## 🛠️ Tools Available

### Built-in Tools

The agents have access to the following tools:

- **File Operations**:
  - `read_file_tool`: Read file contents
  - `write_file_tool`: Write to files
  - `list_files_tool`: List directory contents

- **Code Execution**:
  - `execute_python_code`: Execute code snippets safely
  - `execute_file`: Execute Python files with timeout

- **Analysis**:
  - `analyze_error`: Parse and suggest fixes for errors
  - `search_code`: Search for terms in codebase

### MCP Tool Integration

When enabled (`MCP_ENABLED=true`), agents automatically gain access to:

- **Filesystem MCP Server**: File system operations
- **Custom MCP Servers**: Any MCP-compatible tool server
- **HTTP/STDIO Transports**: Flexible connectivity options

```bash
# Enable MCP in .env
MCP_ENABLED=true
MCP_SERVER_URL=http://localhost:8000/mcp
```

Agents will automatically:
1. Connect to configured MCP servers
2. Retrieve available tools
3. Bind tools to LLM for use
4. Execute tools as needed

Reference: [LangChain MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)

## 🔧 Configuration

Edit `config/llm_config.py` to customize:

- Default model providers
- Model mappings
- Execution timeouts
- MCP settings

```python
# Example configuration
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "gpt-4o"
MAX_EXECUTION_ATTEMPTS = 5
EXECUTION_TIMEOUT = 30  # seconds

# OpenRouter configuration (pre-configured)
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

## 📊 Supported LLM Providers

| Provider | Models | API Key Required | Default |
|----------|--------|------------------|---------|
| OpenRouter | GPT-4o, GPT-3.5 Turbo, Many others | Yes | ✅ |
| Anthropic | Claude 3.5 Sonnet, Opus, Haiku | Yes | |
| OpenAI | GPT-4, GPT-3.5 Turbo | Yes | |
| Google | Gemini 1.5 Pro, Flash | Yes | |
| Ollama | Llama 3.1, Custom models | No (local) | |

## 🎯 Example Use Cases

### 1. Fix Syntax Errors

```bash
python main.py --file buggy_script.py
```

The system will:
- Detect syntax errors
- Suggest and apply fixes
- Verify the code runs

### 2. Add Missing Imports

```bash
python main.py --file incomplete_code.py
```

The system will:
- Identify missing modules
- Add required imports
- Test execution

### 3. Debug Runtime Errors

```bash
python main.py --file error_prone.py
```

The system will:
- Execute the code
- Capture runtime errors
- Apply fixes iteratively

### 4. Run Examples

```bash
python example_usage.py
```

View various usage examples including:
- Basic LLM initialization
- Tool definitions
- Workflow usage
- Configuration validation

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
