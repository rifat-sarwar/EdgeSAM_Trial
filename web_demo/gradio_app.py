# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).

import torch
import gradio as gr
import numpy as np
from edge_sam import sam_model_registry, SamPredictor
from edge_sam.onnx import SamPredictorONNX
from PIL import ImageDraw
from utils.tools_gradio import fast_process
import copy
import argparse

parser = argparse.ArgumentParser(
    description="Host EdgeSAM as a local web service."
)
parser.add_argument(
    "--checkpoint",
    default="weights/edge_sam_3x.pth",
    type=str,
    help="The path to the PyTorch checkpoint of EdgeSAM."
)
parser.add_argument(
    "--encoder-onnx-path",
    default="weights/edge_sam_3x_encoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's encoder."
)
parser.add_argument(
    "--decoder-onnx-path",
    default="weights/edge_sam_3x_decoder.onnx",
    type=str,
    help="The path to the ONNX model of EdgeSAM's decoder."
)
parser.add_argument(
    "--enable-onnx",
    action="store_true",
    help="Use ONNX to speed up the inference.",
)
parser.add_argument(
    "--server-name",
    default="0.0.0.0",
    type=str,
    help="The server address that this demo will be hosted on."
)
parser.add_argument(
    "--port",
    default=8080,
    type=int,
    help="The port that this demo will be hosted on."
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.enable_onnx:
    # device = "cpu"
    predictor = SamPredictorONNX(args.encoder_onnx_path, args.decoder_onnx_path)
else:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bicubic")
    sam = sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)


examples = [
    ["web_demo/assets/1.jpeg"],
    ["web_demo/assets/2.jpeg"],
    ["web_demo/assets/3.jpeg"],
    ["web_demo/assets/4.jpeg"],
    ["web_demo/assets/5.jpeg"],
    ["web_demo/assets/6.jpeg"],
    ["web_demo/assets/7.jpeg"],
    ["web_demo/assets/8.jpeg"],
    ["web_demo/assets/9.jpeg"],
    ["web_demo/assets/10.jpeg"],
    ["web_demo/assets/11.jpeg"],
    ["web_demo/assets/12.jpeg"],
    ["web_demo/assets/13.jpeg"],
    ["web_demo/assets/14.jpeg"],
    ["web_demo/assets/15.jpeg"],
    ["web_demo/assets/16.jpeg"]
]

# Description
title = """
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 2rem;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300; letter-spacing: 2px;">🛍️ SnapSell</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">AI-Powered Product Photo Editor for E-commerce</p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">Segment, edit, and optimize your product images for online selling</p>
</div>
"""

description_p = """
<div class="point-selection-text" style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; margin: 1rem 0;">
    <h3 style="margin: 0 0 1rem 0; color: #000000 !important; font-size: 1.2rem; font-weight: bold; text-shadow: none;">🎯 Point Selection Mode</h3>
    <ol style="margin: 0; padding-left: 1.5rem; color: #000000 !important; line-height: 1.6; font-weight: 600; text-shadow: none;">
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Upload</strong> your product image</li>
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Choose</strong> point type (Positive to include, Negative to exclude)</li>
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Click</strong> on the product to segment it from background</li>
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Edit</strong> your segmented image and post to e-commerce</li>
    </ol>
    <div style="margin-top: 1rem; padding: 0.8rem; background: #e3f2fd; border-radius: 8px; border-left: 3px solid #2196f3;">
        <strong style="color: #000000 !important; font-weight: bold; text-shadow: none;">💡 Pro Tip:</strong> <span style="color: #000000 !important; font-weight: 600; text-shadow: none;">Use positive points on your product, negative points on background areas you want to remove.</span>
    </div>
</div>
"""

description_b = """
<div class="box-selection-text" style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin: 1rem 0;">
    <h3 style="margin: 0 0 1rem 0; color: #000000 !important; font-size: 1.2rem; font-weight: bold; text-shadow: none;">📦 Box Selection Mode</h3>
    <ol style="margin: 0; padding-left: 1.5rem; color: #000000 !important; line-height: 1.6; font-weight: 600; text-shadow: none;">
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Upload</strong> your product image</li>
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Click twice</strong> to create a bounding box around your product</li>
        <li style="color: #000000 !important; font-weight: 600;"><strong style="color: #000000 !important; font-weight: bold;">Edit</strong> your segmented image and post to e-commerce</li>
    </ol>
    <div style="margin-top: 1rem; padding: 0.8rem; background: #e8f5e8; border-radius: 8px; border-left: 3px solid #4caf50;">
        <strong style="color: #000000 !important; font-weight: bold; text-shadow: none;">💡 Pro Tip:</strong> <span style="color: #000000 !important; font-weight: 600; text-shadow: none;">Perfect for rectangular products like books, boxes, or electronics.</span>
    </div>
</div>
"""

css = """
/* SnapSell E-commerce Styling */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
    background: white !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
}

/* Header styling */
h1, h2, h3 {
    font-weight: 300 !important;
    color: #333 !important;
}

/* Override for Point Selection Mode text */
.point-selection-text h3 {
    color: #000000 !important;
    font-weight: bold !important;
    text-shadow: none !important;
}

.point-selection-text ol {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

.point-selection-text li {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

.point-selection-text strong {
    color: #000000 !important;
    font-weight: bold !important;
    text-shadow: none !important;
}

.point-selection-text span {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

/* Force all text in point selection to be black */
.point-selection-text * {
    color: #000000 !important;
    text-shadow: none !important;
}

/* Override for Box Selection Mode text */
.box-selection-text h3 {
    color: #000000 !important;
    font-weight: bold !important;
    text-shadow: none !important;
}

.box-selection-text ol {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

.box-selection-text li {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

.box-selection-text strong {
    color: #000000 !important;
    font-weight: bold !important;
    text-shadow: none !important;
}

.box-selection-text span {
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
}

/* Force all text in box selection to be black */
.box-selection-text * {
    color: #000000 !important;
    text-shadow: none !important;
}

/* Tab styling */
.tab-nav {
    background: #f8f9fa !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    margin-bottom: 2rem !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    padding: 1rem 2rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    background: white !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    color: #667eea !important;
}

/* Button styling */
button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
}

button.secondary {
    background: #6c757d !important;
}

button.secondary:hover {
    background: #5a6268 !important;
    box-shadow: 0 8px 20px rgba(108, 117, 125, 0.4) !important;
}

/* Image container styling */
.image-container {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    border: 3px solid #f8f9fa !important;
}

/* Radio button styling */
.radio-group {
    background: #f8f9fa !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    border: 2px solid #e9ecef !important;
}

/* E-commerce workflow styling */
.workflow-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    padding: 2rem !important;
    border-radius: 15px !important;
    margin: 2rem 0 !important;
    text-align: center !important;
}

.workflow-steps {
    display: flex !important;
    justify-content: space-around !important;
    margin-top: 1.5rem !important;
    flex-wrap: wrap !important;
}

.workflow-step {
    background: rgba(255,255,255,0.2) !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    margin: 0.5rem !important;
    min-width: 150px !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
        margin: 0.5rem !important;
    }
    
    h1 {
        font-size: 1.8rem !important;
    }
    
    .workflow-steps {
        flex-direction: column !important;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #667eea;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #5a6fd8;
}
"""


def reset(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['ori_image'] = None
    session_state['image_with_prompt'] = None
    session_state['feature'] = None
    return None, session_state


def reset_all(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['ori_image'] = None
    session_state['image_with_prompt'] = None
    session_state['feature'] = None
    return None, None, session_state


def clear(session_state):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []
    session_state['image_with_prompt'] = copy.deepcopy(session_state['ori_image'])
    return session_state['ori_image'], session_state


def on_image_upload(
    image,
    session_state,
    input_size=1024
):
    session_state['coord_list'] = []
    session_state['label_list'] = []
    session_state['box_list'] = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    session_state['ori_image'] = copy.deepcopy(image)
    session_state['image_with_prompt'] = copy.deepcopy(image)
    print("Image changed")
    nd_image = np.array(image)
    session_state['feature'] = predictor.set_image(nd_image)

    return image, session_state


def convert_box(xyxy):
    min_x = min(xyxy[0][0], xyxy[1][0])
    max_x = max(xyxy[0][0], xyxy[1][0])
    min_y = min(xyxy[0][1], xyxy[1][1])
    max_y = max(xyxy[0][1], xyxy[1][1])
    xyxy[0][0] = min_x
    xyxy[1][0] = max_x
    xyxy[0][1] = min_y
    xyxy[1][1] = max_y
    return xyxy


def segment_with_points(
    label,
    session_state,
    evt: gr.SelectData,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=False,
):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    session_state['coord_list'].append([x, y])
    session_state['label_list'].append(1 if label == "Positive" else 0)

    print(f"coord_list: {session_state['coord_list']}")
    print(f"label_list: {session_state['label_list']}")

    draw = ImageDraw.Draw(session_state['image_with_prompt'])
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = session_state['image_with_prompt']

    if args.enable_onnx:
        coord_np = np.array(session_state['coord_list'])[None]
        label_np = np.array(session_state['label_list'])[None]
        masks, scores, _ = predictor.predict(
            features=session_state['feature'],
            point_coords=coord_np,
            point_labels=label_np,
        )
        masks = masks.squeeze(0)
        scores = scores.squeeze(0)
    else:
        coord_np = np.array(session_state['coord_list'])
        label_np = np.array(session_state['label_list'])
        masks, scores, logits = predictor.predict(
            features=session_state['feature'],
            point_coords=coord_np,
            point_labels=label_np,
            num_multimask_outputs=4,
            use_stability_score=True
        )

    print(f'scores: {scores}')
    area = masks.sum(axis=(1, 2))
    print(f'area: {area}')

    annotations = np.expand_dims(masks[scores.argmax()], axis=0)

    seg = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    return seg, session_state


def segment_with_box(
        session_state,
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    if len(session_state['box_list']) == 0:
        session_state['box_list'].append([x, y])
    elif len(session_state['box_list']) == 1:
        session_state['box_list'].append([x, y])
    elif len(session_state['box_list']) == 2:
        session_state['image_with_prompt'] = copy.deepcopy(session_state['ori_image'])
        session_state['box_list'] = [[x, y]]

    print(f"box_list: {session_state['box_list']}")

    draw = ImageDraw.Draw(session_state['image_with_prompt'])
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = session_state['image_with_prompt']

    if len(session_state['box_list']) == 2:
        box = convert_box(session_state['box_list'])
        xy = (box[0][0], box[0][1], box[1][0], box[1][1])
        draw.rectangle(
            xy,
            outline=box_color,
            width=box_outline
        )

        box_np = np.array(box)
        if args.enable_onnx:
            point_coords = box_np.reshape(2, 2)[None]
            point_labels = np.array([2, 3])[None]
            masks, _, _ = predictor.predict(
                features=session_state['feature'],
                point_coords=point_coords,
                point_labels=point_labels,
            )
            annotations = masks[:, 0, :, :]
        else:
            masks, scores, _ = predictor.predict(
                features=session_state['feature'],
                box=box_np,
                num_multimask_outputs=1,
            )
            annotations = masks

        seg = fast_process(
            annotations=annotations,
            image=image,
            device=device,
            scale=(1024 // input_size),
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            withContours=withContours,
        )
        return seg, session_state
    return image, session_state


img_p = gr.Image(label="📸 Upload Your Product Image", type="pil", elem_classes="image-container")
img_b = gr.Image(label="📸 Upload Your Product Image", type="pil", elem_classes="image-container")

with gr.Blocks(css=css, title="SnapSell - AI Product Photo Editor") as demo:
    session_state = gr.State({
        'coord_list': [],
        'label_list': [],
        'box_list': [],
        'ori_image': None,
        'image_with_prompt': None,
        'feature': None
    })

    # Header
    gr.HTML(title)

    # E-commerce Workflow
    gr.HTML("""
    <div class="workflow-container">
        <h2 style="margin: 0 0 1rem 0;">🚀 Your E-commerce Workflow</h2>
        <div class="workflow-steps">
            <div class="workflow-step">
                <h3 style="margin: 0 0 0.5rem 0;">1️⃣ Upload</h3>
                <p style="margin: 0; font-size: 0.9rem;">Upload your product photo</p>
            </div>
            <div class="workflow-step">
                <h3 style="margin: 0 0 0.5rem 0;">2️⃣ Segment</h3>
                <p style="margin: 0; font-size: 0.9rem;">Remove background with AI</p>
            </div>
            <div class="workflow-step">
                <h3 style="margin: 0 0 0.5rem 0;">3️⃣ Edit</h3>
                <p style="margin: 0; font-size: 0.9rem;">Perfect your product image</p>
            </div>
            <div class="workflow-step">
                <h3 style="margin: 0 0 0.5rem 0;">4️⃣ Post</h3>
                <p style="margin: 0; font-size: 0.9rem;">Upload to your store</p>
            </div>
        </div>
    </div>
    """)

    with gr.Tab("🎯 Point Selection", elem_classes="tab-nav") as tab_p:
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=2):
                img_p.render()
            with gr.Column(scale=1):
                with gr.Group(elem_classes="radio-group"):
                    add_or_remove = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="🎯 Selection Type",
                        info="Positive: Include product areas\nNegative: Exclude background areas"
                    )
                
                with gr.Row():
                    clear_btn_p = gr.Button("🗑️ Clear Points", variant="secondary", size="sm")
                    reset_btn_p = gr.Button("🔄 Reset Image", variant="secondary", size="sm")
                
                gr.HTML(description_p)

    with gr.Tab("📦 Box Selection", elem_classes="tab-nav") as tab_b:
        with gr.Row(elem_classes="panel"):
            with gr.Column(scale=2):
                img_b.render()
            with gr.Column(scale=1):
                with gr.Row():
                    clear_btn_b = gr.Button("🗑️ Clear Box", variant="secondary", size="sm")
                    reset_btn_b = gr.Button("🔄 Reset Image", variant="secondary", size="sm")
                
                gr.HTML(description_b)

    # E-commerce Tips
    gr.HTML("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; margin: 2rem 0; border-left: 4px solid #28a745;">
        <h3 style="margin: 0 0 1rem 0; color: #333;">💡 E-commerce Photo Tips</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">📸 Photo Quality</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">Use high-resolution images with good lighting for best results.</p>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🎯 Clean Background</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">Remove cluttered backgrounds to make your product stand out.</p>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🛍️ Product Focus</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">Ensure your product is the main focus of the image.</p>
            </div>
        </div>
    </div>
    """)

    img_p.upload(on_image_upload, [img_p, session_state], [img_p, session_state])
    img_p.select(segment_with_points, [add_or_remove, session_state], [img_p, session_state])

    clear_btn_p.click(clear, [session_state], [img_p, session_state])
    reset_btn_p.click(reset, [session_state], [img_p, session_state])
    tab_p.select(fn=reset_all, inputs=[session_state], outputs=[img_p, img_b, session_state])

    img_b.upload(on_image_upload, [img_b, session_state], [img_b, session_state])
    img_b.select(segment_with_box, [session_state], [img_b, session_state])

    clear_btn_b.click(clear, [session_state], [img_b, session_state])
    reset_btn_b.click(reset, [session_state], [img_b, session_state])
    tab_b.select(fn=reset_all, inputs=[session_state], outputs=[img_p, img_b, session_state])

demo.queue()
demo.launch(
    server_name=args.server_name, 
    server_port=args.port,
    debug=True,  # Enable debug mode for auto-reload
    show_error=True  # Show detailed errors
)