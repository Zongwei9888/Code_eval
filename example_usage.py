"""
Example usage of the Multi-Agent Code Assistant
Demonstrates different ways to use the system
"""
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config import get_llm, validate_config
from workflow import create_workflow, interactive_session
from main import quick_fix


# Example 1: Basic LLM usage with OpenRouter
def example_basic_llm():
    """Example of basic LLM initialization"""
    print("\n" + "="*60)
    print("Example 1: Basic LLM Usage")
    print("="*60)
    
    # Initialize LLM using OpenRouter
    model = ChatOpenAI(
        model="gpt-4o",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Or use the config module
    model = get_llm("openrouter", "default")
    
    print("✅ LLM initialized successfully")
    print(f"Model: {model}")


# Example 2: Define and use tools
def example_tools():
    """Example of defining and using tools"""
    print("\n" + "="*60)
    print("Example 2: Tools Definition")
    print("="*60)
    
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply `a` and `b`."""
        return a * b

    @tool
    def add(a: int, b: int) -> int:
        """Adds `a` and `b`."""
        return a + b

    @tool
    def divide(a: int, b: int) -> float:
        """Divide `a` and `b`."""
        return a / b
    
    # Test tools
    print(f"Add 3 + 4 = {add.invoke({'a': 3, 'b': 4})}")
    print(f"Multiply 5 * 6 = {multiply.invoke({'a': 5, 'b': 6})}")
    print(f"Divide 10 / 2 = {divide.invoke({'a': 10, 'b': 2})}")


# Example 3: Use the workflow to fix a file
def example_workflow():
    """Example of using the code improvement workflow"""
    print("\n" + "="*60)
    print("Example 3: Code Improvement Workflow")
    print("="*60)
    
    # Create a test file with errors
    test_code = """
import os

def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# This will cause an error - missing definition
result = calculate_average([1, 2, 3, 4, 5])
print(f"Result: {result}")
"""
    
    # Write test file
    with open("test_example.py", "w") as f:
        f.write(test_code)
    
    print("Created test file with intentional error")
    print("\nRunning workflow to fix the code...")
    
    # Use quick_fix function
    try:
        result = quick_fix("test_example.py", provider="openrouter", max_attempts=3)
        print(f"\n✅ Workflow completed!")
        print(f"Success: {result.get('execution_success')}")
        print(f"Attempts: {result.get('execution_attempts')}")
    except Exception as e:
        print(f"❌ Error: {e}")


# Example 4: Validate configuration
def example_config():
    """Example of configuration validation"""
    print("\n" + "="*60)
    print("Example 4: Configuration Validation")
    print("="*60)
    
    validate_config()


# Example 5: Interactive session (commented out for automated run)
def example_interactive():
    """Example of interactive session (run manually)"""
    print("\n" + "="*60)
    print("Example 5: Interactive Session")
    print("="*60)
    print("To start interactive session, run:")
    print("  python main.py --interactive")
    print("\nOr in code:")
    print("  from workflow import interactive_session")
    print("  interactive_session(llm_provider='openrouter')")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Multi-Agent Code Assistant - Examples")
    print("="*80)
    
    # Run examples
    example_basic_llm()
    example_tools()
    example_config()
    example_interactive()
    
    # Uncomment to run workflow example (requires API key)
    # example_workflow()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)

