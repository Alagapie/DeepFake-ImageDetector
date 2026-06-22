import os
import sys
from pathlib import Path

MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]

BLOB_ACCOUNT = "xaimagestorage"
BLOB_CONTAINER = "models"
BLOB_ENDPOINT = f"https://{BLOB_ACCOUNT}.blob.core.windows.net"


def ensure_model(model_dir: Path) -> None:
    if all((model_dir / f).exists() for f in MODEL_FILES):
        print(f"[ModelDownloader] All model files present in {model_dir}", flush=True)
        return

    sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    if not sas_token:
        raise RuntimeError(
            "Model files not found and AZURE_STORAGE_SAS_TOKEN env var not set. "
            "Place model files in models/ or set the env var."
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        dest = model_dir / filename
        if dest.exists():
            print(f"[ModelDownloader] {filename} already exists, skipping", flush=True)
            continue

        url = f"{BLOB_ENDPOINT}/{BLOB_CONTAINER}/{filename}?{sas_token}"
        print(f"[ModelDownloader] Downloading {filename}...", flush=True)
        try:
            _download(url, dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"[ModelDownloader] {filename} saved ({size_mb:.1f} MB)", flush=True)
        except Exception as e:
            print(f"[ModelDownloader] FAILED to download {filename}: {e}", flush=True)
            raise


def _download(url: str, dest: Path) -> None:
    import requests
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
