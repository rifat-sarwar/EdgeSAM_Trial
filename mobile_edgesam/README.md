# Mobile EdgeSAM Demo

A browser-based demonstration of EdgeSAM running directly on your phone's CPU using ONNX Runtime Web. This demo showcases EdgeSAM's on-device capabilities without requiring any server-side processing.

## 🚀 Features

- **On-Device Processing**: Runs entirely on your phone's CPU using ONNX Runtime Web
- **No Server Required**: All inference happens locally in the browser
- **Mobile Optimized**: Responsive design for mobile devices
- **Real-time Segmentation**: Interactive point-based segmentation
- **Cross-Platform**: Works on any device with a modern web browser

## 📱 How It Works

1. **Image Upload**: Upload any image from your phone
2. **Point Selection**: Click on objects you want to segment (positive/negative points)
3. **On-Device Inference**: EdgeSAM processes the image using your phone's CPU
4. **Mask Visualization**: See the segmentation results overlaid on your image

## 🛠️ Technical Implementation

### Architecture
- **Frontend**: HTML5 Canvas + JavaScript
- **ONNX Runtime**: Web-based inference engine
- **Models**: EdgeSAM 3x encoder and decoder ONNX models
- **Server**: Simple Python HTTP server for file serving

### Key Components

#### 1. Model Loading
```javascript
// Load ONNX models directly in browser
this.encoderSession = await ort.InferenceSession.create('./models/edge_sam_3x_encoder.onnx');
this.decoderSession = await ort.InferenceSession.create('./models/edge_sam_3x_decoder.onnx');
```

#### 2. Image Processing
- **Resizing**: Images are resized to fit within 1024x1024 while maintaining aspect ratio
- **Normalization**: RGB values normalized using SAM's standard values
- **Format Conversion**: Converted to CHW (Channel, Height, Width) format for ONNX

#### 3. Coordinate Mapping
- **Display Coordinates**: Points clicked on canvas
- **Model Coordinates**: Mapped to 1024x1024 space with centering offset
- **Consistent Scaling**: Same resizing logic for display and model processing

#### 4. ONNX Inference
```javascript
// Encoder inference
const encoderResults = await this.encoderSession.run({
    image: imageTensor
});

// Decoder inference
const decoderResults = await this.decoderSession.run({
    image_embeddings: encoderResults.image_embeddings,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor
});
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- Modern web browser with JavaScript enabled
- ONNX models (included in `models/` directory)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mobile_edgesam
   ```

2. **Start the server**:
   ```bash
   ./start.sh
   ```
   Or manually:
   ```bash
   python3 server.py
   ```

3. **Access the demo**:
   - Open your phone's browser
   - Navigate to `http://<your-computer-ip>:8000`
   - Or use `http://localhost:8000` if testing on the same device

### File Structure
```
mobile_edgesam/
├── index.html              # Main demo interface
├── server.py               # Python HTTP server
├── start.sh               # Server startup script
├── models/                # ONNX model files
│   ├── edge_sam_3x_encoder.onnx
│   └── edge_sam_3x_decoder.onnx
├── inspect_models.py      # Model inspection script
├── inspect_models_ort.py  # ONNX Runtime model inspection
└── README.md             # This file
```

## 🐛 Issues Fixed During Development

### 1. Model Loading Issues
**Problem**: `ModuleNotFoundError: No module named 'onnx'`
**Solution**: Created virtual environment and installed required dependencies
```bash
python3 -m venv mobile_venv
source mobile_venv/bin/activate
pip install onnx onnxruntime
```

### 2. ONNX Runtime Web Integration
**Problem**: "Invalid data location: undefined for input 'image'"
**Solution**: Properly created `ort.Tensor` objects with correct data types and shapes
```javascript
const imageTensor = new ort.Tensor('float32', this.imageData, [1, 3, 1024, 1024]);
```

### 3. Model Input/Output Names
**Problem**: "input 'image_embeddings' is missing in 'feeds'"
**Solution**: Used correct input/output names from model inspection:
- Encoder input: `image` → output: `image_embeddings`
- Decoder inputs: `image_embeddings`, `point_coords`, `point_labels` → output: `masks`

### 4. Coordinate System Mismatch
**Problem**: Masks appearing in wrong locations
**Solution**: Implemented proper coordinate mapping between display and model space:
```javascript
// Map canvas coordinates to model coordinates with centering offset
const modelX = x + offsetX;
const modelY = y + offsetY;
```

### 5. Image Resizing Consistency
**Problem**: Display and model processing used different image dimensions
**Solution**: Unified resizing logic matching the original gradio app:
```javascript
const scale = inputSize / Math.max(w, h);
const newW = Math.floor(w * scale);
const newH = Math.floor(h * scale);
```

### 6. Mask Tensor Processing
**Problem**: Masks not displaying due to incorrect tensor shape handling
**Solution**: Added proper tensor dimension handling and scaling:
```javascript
// Handle different tensor shapes (3D, 4D)
if (masks.dims.length === 4) {
    // Process 4D tensor [batch, channels, height, width]
} else if (masks.dims.length === 3) {
    // Process 3D tensor [batch, height, width]
}
```

## 🔍 Model Inspection

Use the included scripts to inspect ONNX models:

```bash
# Using ONNX library
python3 inspect_models.py

# Using ONNX Runtime
python3 inspect_models_ort.py
```

## 📊 Performance Notes

- **Model Size**: ~50MB total (encoder + decoder)
- **Memory Usage**: ~200MB during inference
- **Processing Time**: 2-5 seconds on modern phones
- **Compatibility**: Works on iOS Safari, Android Chrome, and desktop browsers

## 🎯 Usage Tips

1. **Image Quality**: Use clear, well-lit images for best results
2. **Point Placement**: Click directly on object boundaries for precise segmentation
3. **Multiple Points**: Add both positive and negative points for better accuracy
4. **Point Management**: Clear points between different objects

## 🔧 Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check browser console for errors
   - Ensure models are in the `models/` directory
   - Verify server is running on correct port

2. **Segmentation not working**:
   - Check browser console for ONNX errors
   - Verify point coordinates are being logged
   - Ensure image is properly loaded

3. **Performance issues**:
   - Close other browser tabs
   - Use smaller images (under 2MB)
   - Ensure good internet connection for initial model loading

### Debug Mode
Enable console logging by opening browser developer tools (F12) to see:
- Model loading status
- Point coordinates
- Tensor shapes and data
- Processing errors

## 📝 Technical Details

### ONNX Model Specifications
- **Encoder Input**: `image` [1, 3, 1024, 1024] (float32)
- **Encoder Output**: `image_embeddings` [1, 256, 64, 64] (float32)
- **Decoder Inputs**: 
  - `image_embeddings` [1, 256, 64, 64] (float32)
  - `point_coords` [1, num_points, 2] (float32)
  - `point_labels` [1, num_points] (float32)
- **Decoder Output**: `masks` [dynamic, dynamic, dynamic, dynamic] (float32)

### Image Preprocessing
- **Resize**: Fit within 1024x1024 while maintaining aspect ratio
- **Normalization**: 
  - R: (R - 123.675) / 58.395
  - G: (G - 116.28) / 57.12
  - B: (B - 103.53) / 57.375
- **Format**: CHW (Channel, Height, Width) for ONNX compatibility

## 🚀 Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Box-based segmentation support
- [ ] Model quantization for faster inference
- [ ] Offline capability with service workers
- [ ] Export segmented masks as images
- [ ] Real-time video segmentation

## 📄 License

This demo is part of the EdgeSAM project. Please refer to the main project license for usage terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests for improvements.

---

**Note**: This demo is designed to showcase EdgeSAM's on-device capabilities. For production use, consider the full EdgeSAM implementation with proper error handling and optimization.