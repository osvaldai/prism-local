#!/usr/bin/env python3
"""Download Gemma model for PRISM Local benchmark."""
import sys, time
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CANDIDATES = [
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-1b-it-4bit",
]

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} → {local_dir}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir),
                          ignore_patterns=["*.bin", "original/*"])
        return True
    except Exception as e:
        print(f"  Failed: {e}"); return False

def verify_model(model_path):
    print(f"Verifying {model_path.name}...")
    try:
        from mlx_lm import load, generate
        model, tokenizer = load(str(model_path))
        out = generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)
        print(f"  OK: '{str(out)[:40]}'"); return True
    except Exception as e:
        print(f"  Failed: {e}"); return False

def main():
    t0 = time.time()
    for repo_id in CANDIDATES:
        name = repo_id.split("/")[-1]
        local_dir = MODELS_DIR / name
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"Already exists: {local_dir.name}")
            if verify_model(local_dir):
                print(f"Model ready: {local_dir}"); return 0
        if download_model(repo_id, local_dir):
            if verify_model(local_dir):
                elapsed = time.time() - t0
                size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1024**2
                print(f"\nDone in {elapsed:.1f}s | Size: {size_mb:.0f} MB")
                print(f"Model path: {local_dir}"); return 0
    print("ERROR: all candidates failed"); return 1

if __name__ == "__main__":
    sys.exit(main())
