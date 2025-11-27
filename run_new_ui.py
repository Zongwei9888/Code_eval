#!/usr/bin/env python
"""
Code Eval v4.0 - Launch Script
Run the NiceGUI-based modern UI
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from new_ui import run_app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Eval v4.0 - NiceGUI Edition")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    run_app(host=args.host, port=args.port, reload=args.reload)
