"""
Main entry point for Multi-Agent Code Assistant
Provides CLI interface for the code improvement workflow
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from workflow import create_workflow, interactive_session
from config import MODEL_MAPPINGS, validate_config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Code Assistant - Automatically analyze, execute, and fix code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze and fix a specific file
  python main.py --file script.py
  
  # Use a different LLM provider
  python main.py --file script.py --provider openai
  
  # Set maximum attempts
  python main.py --file script.py --max-attempts 10
  
  # Interactive mode
  python main.py --interactive
  
  # Use fast model
  python main.py --file script.py --model-type fast
  
  # Validate configuration
  python main.py --validate-config
        """
    )
    
    # File input
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to Python file to analyze and fix"
    )
    
    # Interactive mode
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive session"
    )
    
    # LLM configuration
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=list(MODEL_MAPPINGS.keys()),
        default="openrouter",
        help="LLM provider to use (default: openrouter)"
    )
    
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        choices=["default", "fast", "powerful"],
        default="default",
        help="Model type to use (default: default)"
    )
    
    # Workflow configuration
    parser.add_argument(
        "--max-attempts", "-a",
        type=int,
        default=5,
        help="Maximum execution attempts (default: 5)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream workflow updates in real-time"
    )
    
    parser.add_argument(
        "--visualize",
        type=str,
        metavar="OUTPUT_PATH",
        help="Generate workflow graph visualization"
    )
    
    # Configuration validation
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate and display configuration"
    )
    
    args = parser.parse_args()
    
    # Validate configuration if requested
    if args.validate_config:
        validate_config()
        if not args.file and not args.interactive:
            return
    
    # Validate arguments
    if not args.interactive and not args.file and not args.visualize and not args.validate_config:
        parser.print_help()
        print("\n❌ Error: Please specify --file, --interactive, --visualize, or --validate-config")
        sys.exit(1)
    
    # Create workflow
    try:
        workflow = create_workflow(
            llm_provider=args.provider,
            max_attempts=args.max_attempts
        )
        
        # Visualize workflow
        if args.visualize:
            print(f"📊 Generating workflow visualization...")
            workflow.visualize(args.visualize)
            if not args.file and not args.interactive:
                return
        
        # Interactive mode
        if args.interactive:
            interactive_session(llm_provider=args.provider)
            return
        
        # File mode
        if args.file:
            file_path = Path(args.file)
            
            # Check if file exists
            if not file_path.exists():
                print(f"❌ Error: File not found: {args.file}")
                sys.exit(1)
            
            # Read file content
            try:
                initial_code = file_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"❌ Error reading file: {str(e)}")
                sys.exit(1)
            
            # Run workflow
            if args.stream:
                print("🔄 Streaming workflow updates...")
                for update in workflow.stream_run(str(file_path), initial_code):
                    if args.verbose:
                        print(f"\n📦 State update: {list(update.keys())}")
            else:
                result = workflow.run(str(file_path), initial_code)
                
                # Print final result
                print("\n" + "="*80)
                print("📊 FINAL RESULTS")
                print("="*80)
                print(f"\nFile: {args.file}")
                print(f"Provider: {args.provider}")
                print(f"Attempts: {result.get('execution_attempts', 0)}")
                print(f"Success: {'✅ Yes' if result.get('execution_success') else '❌ No'}")
                
                if result.get('final_status'):
                    print(f"\nStatus: {result['final_status']}")
                
                if args.verbose:
                    print(f"\nModifications: {len(result.get('modification_history', []))}")
                    print(f"Messages: {len(result.get('messages', []))}")
                
                print("="*80)
                
                # Exit code based on success
                sys.exit(0 if result.get('execution_success') else 1)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def quick_fix(file_path: str, provider: str = "openrouter", max_attempts: int = 5) -> dict:
    """
    Quick fix function for programmatic use
    
    Args:
        file_path: Path to file to fix
        provider: LLM provider
        max_attempts: Maximum attempts
        
    Returns:
        Result dictionary
    """
    workflow = create_workflow(llm_provider=provider, max_attempts=max_attempts)
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    initial_code = path.read_text(encoding='utf-8')
    result = workflow.run(str(file_path), initial_code)
    
    return result


if __name__ == "__main__":
    main()

