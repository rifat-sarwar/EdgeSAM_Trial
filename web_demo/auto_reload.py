#!/usr/bin/env python3
"""
Auto-reload script for Gradio app using watchdog
Run this script to automatically restart the app when files change
"""

import subprocess
import sys
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GradioReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_app()
    
    def start_app(self):
        """Start the Gradio app"""
        if self.process:
            self.process.terminate()
            self.process.wait()
        
        print("🚀 Starting Gradio app...")
        self.process = subprocess.Popen([
            sys.executable, "gradio_app.py"
        ])
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        # Only reload for Python files
        if event.src_path.endswith('.py'):
            print(f"📝 File changed: {event.src_path}")
            print("🔄 Reloading app...")
            self.start_app()
    
    def stop(self):
        """Stop the app"""
        if self.process:
            self.process.terminate()
            self.process.wait()

if __name__ == "__main__":
    try:
        # Install watchdog if not available
        try:
            import watchdog
        except ImportError:
            print("📦 Installing watchdog...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
        
        # Start the file watcher
        event_handler = GradioReloadHandler()
        observer = Observer()
        observer.schedule(event_handler, path=".", recursive=True)
        observer.start()
        
        print("👀 Watching for file changes...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            observer.stop()
            event_handler.stop()
        
        observer.join()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
