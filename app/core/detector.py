"""
Detection Engine
Loads the ViT model from ./models/ and runs REAL/FAKE inference.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import DEVICE, IMAGE_SIZE, LABEL_FAKE, LABEL_REAL, MODEL_PATH


class FakeImageDetector:
    _instance: "FakeImageDetector | None" = None

    def __init__(self) -> None:
        print(f"[Detector] Loading model from {MODEL_PATH} on {DEVICE}…")
        self.processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
        self.model = AutoModelForImageClassification.from_pretrained(
    str(MODEL_PATH),
    attn_implementation="eager"
)
        self.model.to(DEVICE)
        self.model.eval()

        # Verify label mapping — log it so you can confirm REAL/FAKE order
        print(f"[Detector] id2label: {self.model.config.id2label}")
        self._validate_labels()
        print("[Detector] Ready.")

    # ── Singleton so model is only loaded once ────────────────────────────────
    @classmethod
    def get(cls) -> "FakeImageDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Label safety check ────────────────────────────────────────────────────
    def _validate_labels(self) -> None:
        labels = list(self.model.config.id2label.values())
        upper = [l.upper() for l in labels]
        if LABEL_FAKE not in upper or LABEL_REAL not in upper:
            raise RuntimeError(
                f"Model labels {labels} don't contain expected REAL/FAKE. "
                f"Update LABEL_FAKE / LABEL_REAL in config.py to match."
            )

    # ── Core prediction ───────────────────────────────────────────────────────
    def predict(self, image: Image.Image) -> dict:
        """
        Returns:
            {
                "label": "FAKE" | "REAL",
                "confidence": 96.4,          # percentage
                "probabilities": {"FAKE": 96.4, "REAL": 3.6}
            }
        """
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        label_id = probs.argmax().item()
        raw_label = self.model.config.id2label[label_id].upper()
        confidence = round(probs[label_id].item() * 100, 2)

        probabilities = {
            self.model.config.id2label[i].upper(): round(p.item() * 100, 2)
            for i, p in enumerate(probs)
        }

        return {
            "label": raw_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    # ── Expose model + processor for explainability modules ───────────────────
    @property
    def vit_model(self):
        return self.model

    @property
    def vit_processor(self):
        return self.processor
