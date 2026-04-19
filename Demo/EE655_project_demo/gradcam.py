import tensorflow as tf
import torch
import numpy as np
import cv2

# =========================================
# KERAS GRAD-CAM (For Stage 1 & Stage 2)
# =========================================
import keras  # add this

def get_keras_gradcam(model, img_array, last_conv_layer_name):
    grad_model = keras.models.Model(       # was tf.keras.models.Model
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)
    
    return heatmap.numpy()

# =========================================
# PYTORCH GRAD-CAM (For Stage 3 YModel)
# =========================================
class PyTorchYModelGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        target_layer = self.model.backbone[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_task="ripeness"):
        self.model.eval()
        self.model.zero_grad()
        
        # Force PyTorch to track gradients
        input_tensor.requires_grad_(True)
        
        fruit_logits, ripe_logits = self.model(input_tensor)
        
        if target_task == "ripeness":
            target_idx = ripe_logits.argmax(dim=1).item()
            score = ripe_logits[0, target_idx]
        else:
            target_idx = fruit_logits.argmax(dim=1).item()
            score = fruit_logits[0, target_idx]
            
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Backward hooks did not fire.")

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) == 0:
            return heatmap
            
        heatmap = heatmap / np.max(heatmap)
        return heatmap