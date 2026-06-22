# XAI Fake Image Detection System

Explainable AI system that classifies images as REAL or FAKE using a Vision Transformer,
with attention maps, Grad-CAM visualizations, metadata analysis, and forensic PDF reports.

---

## Project Structure

```
xai_fake_detector/
├── models/                  ← PUT YOUR DOWNLOADED MODEL FILES HERE
├── uploads/                 ← Images uploaded for analysis (auto-created)
├── reports/                 ← Generated heatmaps + PDF reports (auto-created)
├── app/
│   ├── config.py            ← Central settings (model path, device, etc.)
│   ├── main.py              ← FastAPI application
│   ├── core/
│   │   ├── detector.py      ← ViT model loader + inference
│   │   └── pipeline.py      ← Orchestrates all phases end-to-end
│   ├── explainability/
│   │   ├── attention.py     ← Attention Rollout heatmaps
│   │   └── gradcam.py       ← Grad-CAM for ViT
│   ├── metadata/
│   │   └── analyser.py      ← EXIF metadata inspection
│   ├── evidence/
│   │   └── engine.py        ← Evidence aggregation + NL explanation
│   ├── reporting/
│   │   └── pdf_generator.py ← Forensic PDF generation
│   └── api/
│       └── routes.py        ← FastAPI endpoints
├── test_image.py            ← CLI test (no server needed)
├── requirements.txt
└── README.md
```

---

## Setup (Windows)

### Step 1 — Place your model files

Copy the downloaded model folder contents into `./models/`.
The folder must contain at minimum:
```
models/
├── config.json
├── preprocessor_config.json   (or processor_config.json)
└── pytorch_model.bin          (or model.safetensors)
```

### Step 2 — Create and activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

> If you have a CUDA GPU and want GPU acceleration:
> Install the correct torch+cuda version FIRST from https://pytorch.org/get-started/locally/
> then run the requirements install.

### Step 4 — Verify model labels (important)

Run this one-liner to check the model's label mapping:

```cmd
python -c "from transformers import AutoModelForImageClassification; m = AutoModelForImageClassification.from_pretrained('./models'); print(m.config.id2label)"
```

Expected output: `{0: 'FAKE', 1: 'REAL'}` or `{0: 'REAL', 1: 'FAKE'}`.

If the labels use different strings (e.g. `artificial`, `genuine`), update
`LABEL_FAKE` and `LABEL_REAL` in `app/config.py` to match.

---

## Usage

### Option A — CLI (quickest test)

```cmd
python test_image.py path\to\your\image.jpg
```

This runs the full pipeline and prints results to the terminal.
The PDF report is saved to `./reports/`.

### Option B — FastAPI Server

Start the server:
```cmd
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Interactive docs:** http://localhost:8000/docs

**Analyse an image (curl):**
```cmd
curl -X POST http://localhost:8000/analyze -F "file=@path\to\image.jpg"
```

**Download the PDF:**
```
GET http://localhost:8000/report/<analysis_id>
```

---

## API Response Example

```json
{
  "analysis_id": "a3f9c12b4e7d",
  "filename": "portrait.jpg",
  "timestamp": "2024-11-15 14:32:01 UTC",
  "prediction": "FAKE",
  "confidence": 96.4,
  "probabilities": {"FAKE": 96.4, "REAL": 3.6},
  "attention_regions": ["central region", "upper region"],
  "gradcam_regions": ["central-middle region"],
  "metadata_findings": [
    {"flag": "NO_CAMERA", "detail": "No camera make/model found in metadata", "severity": "high"},
    {"flag": "EDITING_SOFTWARE", "detail": "Image editing software detected: Adobe Photoshop", "severity": "medium"}
  ],
  "evidence_items": [...],
  "nl_explanation": "The image was classified as AI-generated or manipulated with high confidence (96.4%)...",
  "pdf_download_url": "/report/a3f9c12b4e7d"
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: Model labels don't contain REAL/FAKE` | Run the label check command above and update `config.py` |
| `Could not locate target layer` in Grad-CAM | Open `gradcam.py`, run `print(list(model.named_modules()))` and find the last encoder layer name, then update `_get_target_layer()` |
| Slow first request | Normal — model loads on startup. Subsequent requests are fast. |
| GPU not detected | Install CUDA-enabled torch before requirements. Check with `python -c "import torch; print(torch.cuda.is_available())"` |
| `piexif` errors on PNG files | Expected — PNGs don't always have EXIF. The system handles this gracefully. |
