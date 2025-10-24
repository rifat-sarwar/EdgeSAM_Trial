#!/bin/bash
# Development startup script with auto-reload

echo "🚀 Starting EdgeSAM Web Demo in Development Mode"
echo "🔄 Auto-reload enabled - changes will be detected automatically"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Install watchdog if not available
echo "📦 Checking dependencies..."
python3 -c "import watchdog" 2>/dev/null || {
    echo "📦 Installing watchdog for file monitoring..."
    pip3 install watchdog
}

# Start the app with auto-reload
echo "🔧 Starting development server..."
python3 auto_reload.py
