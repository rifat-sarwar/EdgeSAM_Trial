#!/usr/bin/env python3
"""
Simple HTTP server for EdgeSAM Mobile Demo
Serves the mobile web app that runs ONNX models directly in the browser
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Configuration
PORT = 8080
HOST = "0.0.0.0"  # Allow access from other devices on the network

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def main():
    # Change to the directory containing the mobile demo
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    print("🚀 Starting EdgeSAM Mobile Demo Server")
    print(f"📱 Serving on: http://{HOST}:{PORT}")
    print(f"📁 Serving from: {demo_dir}")
    print("\n📋 Instructions:")
    print("1. Open your phone's browser")
    print("2. Go to: http://[YOUR_PC_IP]:8080")
    print("3. Upload an image and start segmenting!")
    print("\n💡 This runs EdgeSAM directly on your phone's CPU!")
    print("\nPress Ctrl+C to stop the server")
    
    # Create server
    with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()
