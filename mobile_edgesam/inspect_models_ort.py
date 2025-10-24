#!/usr/bin/env python3
"""
Inspect ONNX models using onnxruntime to understand input/output names and shapes
"""

import onnxruntime as ort
import numpy as np

def inspect_model(model_path):
    print(f"\n🔍 Inspecting model: {model_path}")
    print("=" * 50)
    
    try:
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        # Get input information
        print("📥 Inputs:")
        for input_meta in session.get_inputs():
            print(f"  Name: {input_meta.name}")
            print(f"  Type: {input_meta.type}")
            print(f"  Shape: {input_meta.shape}")
            print()
        
        # Get output information
        print("📤 Outputs:")
        for output_meta in session.get_outputs():
            print(f"  Name: {output_meta.name}")
            print(f"  Type: {output_meta.type}")
            print(f"  Shape: {output_meta.shape}")
            print()
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    print("🔍 EdgeSAM ONNX Model Inspector")
    print("=" * 40)
    
    # Inspect encoder model
    inspect_model("models/edge_sam_3x_encoder.onnx")
    
    # Inspect decoder model
    inspect_model("models/edge_sam_3x_decoder.onnx")

if __name__ == "__main__":
    main()
