"""
Attention Map Extraction — Attention Rollout for Vision Transformers.

Produces a heatmap showing which image regions the model attended to
when making its REAL/FAKE decision.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import ATTENTION_ALPHA, DEVICE, IMAGE_SIZE


def _get_grid_size(num_patches: int) -> int:
    """Infer grid size from number of patches (excluding [CLS] token)."""
    size = int(math.isqrt(num_patches))
    if size * size != num_patches:
        raise ValueError(f"Cannot infer square grid from {num_patches} patches.")
    return size


def extract_attention_rollout(
    model,
    processor,
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run attention rollout and return:
        - heatmap_overlay : H×W×3 numpy array (original image + heatmap)
        - raw_mask        : grid_size×grid_size float array (0–1)
        - regions         : list of English region labels ["facial region", …]
    """
    image_rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (1, heads, seq, seq) per layer

    # ── Attention Rollout ─────────────────────────────────────────────────────
    # Start with identity, then multiply through each layer
    num_tokens = attentions[0].size(-1)
    rollout = torch.eye(num_tokens, device=DEVICE)

    for attn in attentions:
        # Average over all heads → (seq, seq)
        attn_avg = attn.squeeze(0).mean(dim=0)
        # Add residual connection (skip connection in ViT)
        attn_aug = attn_avg + torch.eye(num_tokens, device=DEVICE)
        # Row-normalise
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        rollout = attn_aug @ rollout

    # CLS token (index 0) attending to patch tokens (indices 1:)
    cls_attention = rollout[0, 1:].cpu().numpy()
    grid_size = _get_grid_size(len(cls_attention))
    mask = cls_attention.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # ── Overlay ───────────────────────────────────────────────────────────────
    img_np = np.array(image_rgb)
    mask_resized = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(
        (mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (
        (1 - ATTENTION_ALPHA) * img_np + ATTENTION_ALPHA * heatmap_rgb
    ).astype(np.uint8)

    # ── Region labelling ──────────────────────────────────────────────────────
    regions = _label_regions(mask)

    return overlay, mask, regions


def _label_regions(mask: np.ndarray) -> list[str]:
    """
    Divide the mask into thirds (top / middle / bottom) and report
    which zones exceed 60% of the peak attention.
    """
    h = mask.shape[0]
    threshold = mask.max() * 0.6
    zone_names = {0: "upper region", 1: "central region", 2: "lower region"}
    zones = []
    for i, (start, end) in enumerate(
        [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
    ):
        if mask[start:end].mean() >= threshold:
            zones.append(zone_names[i])
    return zones if zones else ["distributed regions"]
