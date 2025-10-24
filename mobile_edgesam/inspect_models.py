#!/usr/bin/env python3
"""
Inspect ONNX models to understand input/output names and shapes
"""

import onnx
import numpy as np

def inspect_model(model_path):
    print(f"\n🔍 Inspecting model: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model
        model = onnx.load(model_path)
        
        # Get input information
        print("📥 Inputs:")
        for input_info in model.graph.input:
            print(f"  Name: {input_info.name}")
            print(f"  Type: {input_info.type}")
            if input_info.type.tensor_type.shape:
                dims = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_info.type.tensor_type.shape.dim]
                print(f"  Shape: {dims}")
            print()
        
        # Get output information
        print("📤 Outputs:")
        for output_info in model.graph.output:
            print(f"  Name: {output_info.name}")
            print(f"  Type: {output_info.type}")
            if output_info.type.tensor_type.shape:
                dims = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_info.type.tensor_type.shape.dim]
                print(f"  Shape: {dims}")
            print()
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    print("🔍 EdgeSAM ONNX Model Inspector (ONNX)")
    print("=" * 40)
    
    # Inspect encoder model
    inspect_model("models/edge_sam_3x_encoder.onnx")
    
    # Inspect decoder model
    inspect_model("models/edge_sam_3x_decoder.onnx")

if __name__ == "__main__":
    main()