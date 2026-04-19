import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import time  # <-- Added for the Speedometer

from gradcam import get_keras_gradcam, PyTorchYModelGradCAM

import keras
from keras import layers as keras_layers

# ── Keras 3 compatibility patch for Lambda layers saved without output_shape ──
_original_compute = keras_layers.Lambda.compute_output_shape

def _patched_compute_output_shape(self, input_shape):
    try:
        return _original_compute(self, input_shape)
    except NotImplementedError:
        return input_shape  # safe fallback — passes input shape through

keras_layers.Lambda.compute_output_shape = _patched_compute_output_shape
# ─────────────────────────────────────────────────────────────────────────────


# =========================================
# 1. DEFINE PYTORCH ARCHITECTURE
# =========================================
class YModel(nn.Module):
    def __init__(self, dropout=0.5, freeze_backbone=True):
        super().__init__()
        efficientnet = models.efficientnet_v2_s(weights=None)
        self.backbone = efficientnet.features
        in_features = 1280
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fruit_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 5),
        )
        self.ripeness_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat).view(x.size(0), -1)
        return self.fruit_head(feat), self.ripeness_head(feat)

# =========================================
# 2. LOAD ALL MODELS & CLASSES
# =========================================
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr=0.001, total_steps=1200.0, warmup_steps=120, min_lr=1e-07, **kwargs):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        return self.min_lr

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

custom_objs = {"WarmupCosineDecay": WarmupCosineDecay}
# ─────────────────────────────────────────────

print("Loading Keras Models...")
model1 = keras.models.load_model(
    "models/ResNet50_FULL_MODEL.keras", 
    custom_objects=custom_objs, 
    safe_mode=False, 
    compile=False
)

model2 = keras.models.load_model(
    "models/ResNet50_V3_FULL_MODEL.keras", 
    custom_objects=custom_objs, 
    safe_mode=False, 
    compile=False
)

print("Loading PyTorch Model...")
device = torch.device("cpu")
pytorch_model = YModel(freeze_backbone=False)
pytorch_model.load_state_dict(torch.load("models/best_ymodel.pth", map_location=device, weights_only=True))
pytorch_model.eval()
pt_cam_extractor = PyTorchYModelGradCAM(pytorch_model)

# Shared Labels
KERAS_CLASSES = sorted([
    "Apple_Unripe", "Apple_Ripe", "Apple_Overripe",
    "Banana_Unripe", "Banana_Ripe", "Banana_Overripe",
    "Mango_Unripe", "Mango_Ripe", "Mango_Overripe",
    "Orange_Unripe", "Orange_Ripe", "Orange_Overripe",
    "Tomato_Unripe", "Tomato_Ripe", "Tomato_Overripe"
])

FRUIT_CLASSES = ["Apple", "Banana", "Mango", "Orange", "Tomato"]
RIPENESS_CLASSES = ["Overripe", "Ripe", "Unripe"]

# =========================================
# 3. PREPROCESSING & UTILS
# =========================================
def preprocess_tf(img):
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = keras.applications.resnet50.preprocess_input(img_resized.astype(np.float32))
    return np.expand_dims(img_preprocessed, axis=0)

def preprocess_pt(img):
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfms(img).unsqueeze(0)

def overlay_heatmap(img, heatmap):
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img.astype("uint8"), 0.6, heatmap_rgb, 0.4, 0)

# =========================================
# 4. ROUTER INFERENCE FUNCTION
# =========================================
def predict(img, model_choice):
    if img is None:
        return None, None, "0.00 ms"

    start_time = time.time()  # <-- Start Speedometer

    if "Stage 1" in model_choice or "Stage 2" in model_choice:
        model = model1 if "Stage 1" in model_choice else model2
        img_tf = preprocess_tf(img)
        
        # Get probabilities for all 15 classes
        preds = model.predict(img_tf, verbose=0)[0]
        
        # Format as a dictionary for Gradio's Label component
        confidences = {KERAS_CLASSES[i].replace("_", " (") + ")": float(preds[i]) for i in range(len(KERAS_CLASSES))}
        heatmap = get_keras_gradcam(model, img_tf, "conv5_block3_out")
        
    else: # Stage 3: PyTorch YModel
        img_pt = preprocess_pt(img)
        with torch.no_grad():
            fruit_out, ripe_out = pytorch_model(img_pt)
            
            # Convert logits to probabilities
            fruit_probs = torch.softmax(fruit_out, dim=1)[0]
            ripe_probs = torch.softmax(ripe_out, dim=1)[0]
            
            # Calculate joint probabilities for all 15 combinations
            confidences = {}
            for f_idx, f_name in enumerate(FRUIT_CLASSES):
                for r_idx, r_name in enumerate(RIPENESS_CLASSES):
                    joint_prob = float(fruit_probs[f_idx] * ripe_probs[r_idx])
                    confidences[f"{f_name} ({r_name})"] = joint_prob
                    
        heatmap = pt_cam_extractor.generate_heatmap(img_pt, target_task="ripeness")

    cam_overlay = overlay_heatmap(img, heatmap)
    
    end_time = time.time()  # <-- End Speedometer
    latency_ms = f"{(end_time - start_time) * 1000:.2f} ms"

    return confidences, cam_overlay, latency_ms

# =========================================
# 5. USER INTERFACE (Gradio)
# =========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🍎 FruitNet: Progressive Quality Assessment")
    gr.Markdown("Select a model stage below to see how the architecture's confidence and attention maps evolve.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Fruit Image")
            
            model_selector = gr.Radio(
                choices=[
                    "Stage 1: Baseline (ResNet-50)", 
                    "Stage 2: Fine-Tuned (ResNet-50)", 
                    "Stage 3: Improved (EfficientNetV2-S)"
                ],
                value="Stage 3: Improved (EfficientNetV2-S)",
                label="Select Model Architecture",
                interactive=True
            )
            
            # Put the EXACT filenames you see in your Colab folder here
            colab_examples = [
                "/content/examples/tomato_overripe_kaggle2_0733.jpg",
                "/content/examples/tomato_unripe_kaggle2_0695.jpg", 
            ]

            gr.Examples(
                examples=colab_examples, 
                inputs=input_img,
                label="Test Examples"
            )
            btn = gr.Button("Analyze Quality", variant="primary")
            
        with gr.Column(scale=1):
            with gr.Row():
                # Replaced generic text with a top-3 Confidence Label and Latency counter
                output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions (Confidence)")
                latency_text = gr.Textbox(label="Inference Latency", lines=1)
            
            gr.Markdown("### Network Attention (Grad-CAM)")
            output_cam = gr.Image(label="Feature Activation Heatmap")

    btn.click(predict, inputs=[input_img, model_selector], outputs=[output_label, output_cam, latency_text])
    model_selector.change(predict, inputs=[input_img, model_selector], outputs=[output_label, output_cam, latency_text])

if __name__ == "__main__":
    demo.launch(share=True)