#!/usr/bin/env python3
"""
Test script to verify EdgeSAM Mobile Demo setup
"""

import os
import sys
from pathlib import Path

def test_setup():
    print("🔍 Testing EdgeSAM Mobile Demo Setup")
    print("=" * 40)
    
    # Check current directory
    current_dir = Path(__file__).parent
    print(f"📁 Current directory: {current_dir}")
    
    # Check required files
    required_files = [
        "index.html",
        "server.py", 
        "models/edge_sam_3x_encoder.onnx",
        "models/edge_sam_3x_decoder.onnx"
    ]
    
    print("\n📋 Checking required files:")
    all_good = True
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Check if we can import required modules
    try:
        import http.server
        import socketserver
        print("  ✅ HTTP server modules available")
    except ImportError as e:
        print(f"  ❌ HTTP server modules: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 Setup looks good! Ready to run EdgeSAM Mobile Demo")
        print("\n🚀 To start the demo:")
        print("   python3 server.py")
        print("   # or")
        print("   ./start.sh")
    else:
        print("❌ Setup incomplete. Please check missing files.")
        return False
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
