#!/usr/bin/env python3
"""PRISM benchmark — measures PRISM-optimized inference vs baseline prompts."""
import json, gc, time
from pathlib import Path
import psutil

MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent

TEST_PROMPTS = [
    ("simple",  "What is 2+2?", 64),
    ("medium",  "Explain how transformers work in machine learning in 3 sentences.", 256),
    ("complex", "Write a detailed technical analysis of KV-cache optimization techniques for large language model inference, covering memory management, sliding window attention, and quantization strategies.", 512),
]

def get_model_path():
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"No model in {MODELS_DIR}")
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            return d
    raise FileNotFoundError(f"No model in {MODELS_DIR}")

def measure_ram():
    return psutil.Process().memory_info().rss / 1024**2

def run_prism_benchmark():
    from mlx_lm import load
    from prism_engine import PRISMEngine, profile_hardware, classify_complexity

    model_path = get_model_path()
    print(f"Model: {model_path.name}")

    t_load = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"Load: {load_ms:.0f} ms | RAM: {measure_ram():.0f} MB")

    hw = profile_hardware()
    engine = PRISMEngine(model, tokenizer, hw)
    engine.print_profile()

    results = []
    for label, prompt, max_tok in TEST_PROMPTS:
        tier = classify_complexity(prompt)
        print(f"\n[{label.upper()}] tier={tier.value} prompt_len={len(prompt)}")
        gc.collect()

        result = engine.generate(prompt)

        print(f"  Tier: {result.tier.value} | Tokens: {result.tokens_generated}")
        print(f"  Time: {result.total_sec:.2f}s | TPS: {result.tokens_per_sec:.1f}")
        print(f"  RAM: {result.ram_mb:.0f} MB")
        print(f"  Context: {result.original_context_len} → {result.compressed_context_len} chars "
              f"({'compressed' if result.context_compressed else 'unchanged'})")
        print(f"  Params: temp={result.params.temperature} top_p={result.params.top_p} "
              f"max_tok={result.params.max_tokens}")
        print(f"  Output: {result.output[:80]}...")

        results.append({
            "label": label,
            "tier": result.tier.value,
            "prompt_len": len(prompt),
            "output_tokens": result.tokens_generated,
            "total_sec": result.total_sec,
            "tokens_per_sec": result.tokens_per_sec,
            "ram_mb": result.ram_mb,
            "context_compressed": result.context_compressed,
            "original_context_len": result.original_context_len,
            "compressed_context_len": result.compressed_context_len,
            "max_tokens_used": result.params.max_tokens,
        })

    out_file = RESULTS_DIR / "prism_results.json"
    with open(out_file, "w") as f:
        json.dump({"model": model_path.name, "results": results, "load_ms": round(load_ms)}, f, indent=2)
    print(f"\nPRISM results saved: {out_file}")
    return results

if __name__ == "__main__":
    run_prism_benchmark()
