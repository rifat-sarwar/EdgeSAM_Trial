#!/bin/bash

echo "🚀 Starting EdgeSAM Mobile Demo"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if ONNX models exist
if [ ! -f "models/edge_sam_3x_encoder.onnx" ] || [ ! -f "models/edge_sam_3x_decoder.onnx" ]; then
    echo "❌ ONNX models not found in models/ directory"
    echo "Please ensure the following files exist:"
    echo "  - models/edge_sam_3x_encoder.onnx"
    echo "  - models/edge_sam_3x_decoder.onnx"
    exit 1
fi

# Get local IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')

echo "📱 Mobile Demo Ready!"
echo "===================="
echo "🌐 Access from your phone: http://$LOCAL_IP:8080"
echo ""
echo "📋 Instructions:"
echo "1. Connect your phone to the same WiFi"
echo "2. Open browser and go to: http://$LOCAL_IP:8080"
echo "3. Upload an image and start segmenting!"
echo ""
echo "💡 This runs EdgeSAM directly on your phone's CPU!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 server.py
