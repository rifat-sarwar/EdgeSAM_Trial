#!/usr/bin/env python3
"""
Run Gradio app in development mode with auto-reload
"""

import os
import sys
import subprocess

def main():
    # Set environment variables for development
    os.environ['GRADIO_DEBUG'] = '1'
    os.environ['GRADIO_DEV_MODE'] = '1'
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("🔧 Starting Gradio app in development mode...")
    print("🔄 Auto-reload enabled")
    print("📝 Watching for file changes...")
    
    try:
        # Run the gradio app with auto-reload
        subprocess.run([
            sys.executable, "-m", "gradio", "gradio_app.py",
            "--reload",  # Enable auto-reload
            "--debug"    # Enable debug mode
        ])
    except KeyboardInterrupt:
        print("\n🛑 Stopping development server...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
