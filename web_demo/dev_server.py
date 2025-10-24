#!/usr/bin/env python3
"""
Development server with auto-reload for Gradio app
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

class GradioDevServer:
    def __init__(self):
        self.process = None
        self.watching = True
        self.last_modified = {}
        self.watch_files = [
            "gradio_app.py",
            "utils/tools_gradio.py",
            "utils/tools.py"
        ]
    
    def get_file_mtime(self, filepath):
        """Get file modification time"""
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return 0
    
    def check_files(self):
        """Check if any watched files have changed"""
        for filepath in self.watch_files:
            if os.path.exists(filepath):
                current_mtime = self.get_file_mtime(filepath)
                if filepath not in self.last_modified:
                    self.last_modified[filepath] = current_mtime
                elif current_mtime > self.last_modified[filepath]:
                    print(f"📝 File changed: {filepath}")
                    return True
        return False
    
    def start_app(self):
        """Start the Gradio app"""
        if self.process:
            print("🛑 Stopping previous app...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        print("🚀 Starting Gradio app...")
        self.process = subprocess.Popen([
            sys.executable, "gradio_app.py"
        ])
    
    def file_watcher(self):
        """File watching thread"""
        while self.watching:
            if self.check_files():
                print("🔄 Reloading app...")
                self.start_app()
                # Update modification times
                for filepath in self.watch_files:
                    if os.path.exists(filepath):
                        self.last_modified[filepath] = self.get_file_mtime(filepath)
            
            time.sleep(1)
    
    def run(self):
        """Run the development server"""
        print("🔧 Starting development server...")
        print("👀 Watching files:", ", ".join(self.watch_files))
        print("Press Ctrl+C to stop")
        
        # Start the app
        self.start_app()
        
        # Start file watcher in separate thread
        watcher_thread = threading.Thread(target=self.file_watcher, daemon=True)
        watcher_thread.start()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping development server...")
            self.watching = False
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()

if __name__ == "__main__":
    server = GradioDevServer()
    server.run()
