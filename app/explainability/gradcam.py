"""
Grad-CAM for Vision Transformers.

Uses pytorch-grad-cam with the ViT reshape transform so the library
can handle the (batch, seq, dim) → (batch, dim, H, W) conversion.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.config import DEVICE, IMAGE_SIZE


def _vit_reshape_transform(tensor, height=None, width=None):
    """
    Convert ViT output from (batch, seq_len, dim) to (batch, dim, H, W).
    Drops the [CLS] token (index 0) and reshapes remaining patch tokens.
    """
    result = tensor[:, 1:, :]          # drop CLS token
    num_patches = result.size(1)
    h = w = int(math.isqrt(num_patches))
    result = result.reshape(result.size(0), h, w, result.size(2))
    result = result.transpose(2, 3).transpose(1, 2)   # → (B, dim, H, W)
    return result

def _get_target_layer(model):
    """
    Return the last transformer encoder block's layernorm_after.
    This model uses vit.layers instead of vit.encoder.layer.
    """
    return [model.vit.layers[-1].layernorm_after]
class _ViTWrapper(torch.nn.Module):
    """
    Wraps the HF ViT model so it returns raw logits as a tensor.
    Grad-CAM requires a plain tensor output, not a ModelOutput object.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def extract_gradcam(
    model,
    processor,
    image: Image.Image,
    target_class_idx: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        - cam_overlay : H×W×3 numpy array (image + CAM heatmap)
        - regions     : list of English region descriptions
    """
    image_rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(image_rgb, dtype=np.float32) / 255.0

    inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)

    # Wrap model so Grad-CAM receives plain logits tensor
    wrapped = _ViTWrapper(model)
    target_layers = _get_target_layer(wrapped.model)
    targets = [ClassifierOutputTarget(target_class_idx)]

    cam = GradCAM(
        model=wrapped,
        target_layers=target_layers,
        reshape_transform=_vit_reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=inputs["pixel_values"],
        targets=targets,
    )[0]  # shape: (H, W)

    cam_overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    regions = _label_gradcam_regions(grayscale_cam)
    return cam_overlay, regions


def _label_gradcam_regions(cam_map: np.ndarray) -> list[str]:
    """
    Threshold the CAM at 50% of max, find contours, and describe
    the most activated areas in plain English.
    """
    thresh = (cam_map > cam_map.max() * 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = cam_map.shape
    descriptions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        cx, cy = x + cw / 2, y + ch / 2
        v = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "central")
        hz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "middle")
        descriptions.append(f"{v}-{hz} region")

    # Deduplicate and cap at 3 regions
    seen, unique = set(), []
    for d in descriptions:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique[:3] if unique else ["central region"]
