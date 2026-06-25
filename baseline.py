#!/usr/bin/env python3
"""Baseline vanilla MLX inference benchmark — no PRISM optimizations."""
import time, json, gc
from pathlib import Path
import psutil

MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent

TEST_PROMPTS = [
    ("simple", "What is 2+2?", 64),
    ("medium", "Explain how transformers work in machine learning in 3 sentences.", 256),
    ("complex", "Write a detailed technical analysis of KV-cache optimization techniques for large language model inference, covering memory management, sliding window attention, and quantization strategies.", 512),
]

def get_model_path():
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"No model found in {MODELS_DIR}")
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            return d
    raise FileNotFoundError(f"No model found in {MODELS_DIR}")

def measure_ram():
    proc = psutil.Process()
    return proc.memory_info().rss / 1024**2

def run_baseline():
    from mlx_lm import load, generate
    model_path = get_model_path()
    print(f"Model: {model_path.name}")
    print(f"RAM before load: {measure_ram():.0f} MB")

    t_load = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_ms = (time.perf_counter() - t_load) * 1000
    ram_after_load = measure_ram()
    print(f"Load time: {load_ms:.0f} ms | RAM after load: {ram_after_load:.0f} MB")

    results = []
    for label, prompt, max_tok in TEST_PROMPTS:
        print(f"\n[{label.upper()}] prompt len={len(prompt)} max_tokens={max_tok}")
        gc.collect()

        # TTFT: time to first token = prefill
        t0 = time.perf_counter()
        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tok,
            verbose=False,
        )
        total_s = time.perf_counter() - t0

        out_tokens = len(tokenizer.encode(output)) if hasattr(tokenizer, "encode") else len(output.split())
        tps = out_tokens / total_s if total_s > 0 else 0
        ram_peak = measure_ram()

        print(f"  Tokens: {out_tokens} | Time: {total_s:.2f}s | TPS: {tps:.1f} | RAM: {ram_peak:.0f} MB")
        print(f"  Output: {output[:80]}...")

        results.append({
            "label": label,
            "prompt_len": len(prompt),
            "max_tokens": max_tok,
            "output_tokens": out_tokens,
            "total_sec": round(total_s, 3),
            "tokens_per_sec": round(tps, 1),
            "ram_mb": round(ram_peak, 0),
        })

    out_file = RESULTS_DIR / "baseline_results.json"
    with open(out_file, "w") as f:
        json.dump({"model": model_path.name, "results": results, "load_ms": round(load_ms)}, f, indent=2)
    print(f"\nBaseline results saved: {out_file}")
    return results

if __name__ == "__main__":
    run_baseline()
