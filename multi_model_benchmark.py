#!/usr/bin/env python3
"""
PRISM Multi-Model Benchmark
Tests PRISM algorithm across multiple model sizes on Apple M1.
"""
import json
import time
import psutil
import platform
from pathlib import Path
from prism_engine_v2 import PRISMEngine, profile_hardware, classify_complexity

MODELS = [
    {
        "id": "mlx-community/gemma-3-4b-it-4bit",
        "name": "Gemma 3 4B INT4",
        "size_gb": 3.3,
        "local_path": "./models/gemma-3-4b-it-4bit",
    },
    {
        "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "name": "Mistral 7B INT4",
        "size_gb": 4.1,
        "local_path": "./models/Mistral-7B-Instruct-v0.3-4bit",
    },
    {
        "id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "name": "Llama 3.1 8B INT4",
        "size_gb": 4.7,
        "local_path": "./models/Meta-Llama-3.1-8B-Instruct-4bit",
    },
]

PROMPTS = [
    {
        "label": "simple",
        "text": "What is 2+2? Give me a short answer.",
    },
    {
        "label": "medium",
        "text": "Explain how transformers work in neural networks, focus on the attention mechanism",
    },
    {
        "label": "complex",
        "text": (
            "Analyze KV-cache optimization techniques for large language model inference "
            "including sliding window attention, quantization approaches, multi-head attention "
            "compression and their tradeoffs for deployment on resource-constrained edge devices"
        ),
    },
]


def check_ram(required_gb: float) -> bool:
    free = psutil.virtual_memory().available / 1024**3
    print(f"  RAM free: {free:.1f} GB | Required: {required_gb:.1f} GB", end="")
    if free < required_gb:
        # macOS Apple Silicon uses NVMe swap + memory compression — attempt anyway
        print(f"  <- LOW RAM (will use swap, expect slower TPS)")
    else:
        print()
    return True  # always attempt, let MLX handle OOM


def load_model(model_cfg: dict):
    from mlx_lm import load
    local = Path(model_cfg["local_path"])
    source = str(local) if local.exists() else model_cfg["id"]
    print(f"  Loading from: {source}")
    t0 = time.perf_counter()
    model, tokenizer = load(source)
    load_ms = int((time.perf_counter() - t0) * 1000)
    return model, tokenizer, load_ms


def benchmark_model(model_cfg: dict) -> dict | None:
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"Model: {name} ({model_cfg['size_gb']} GB INT4)")
    print(f"{'='*60}")

    ram_needed = model_cfg["size_gb"] + 0.8
    if not check_ram(ram_needed):
        return None

    try:
        model, tokenizer, load_ms = load_model(model_cfg)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return None

    hw = profile_hardware()
    engine = PRISMEngine(model, tokenizer, hw)

    results = []
    for p in PROMPTS:
        tier = classify_complexity(p["text"])
        print(f"\n  [{p['label'].upper()}] tier={tier.value} len={len(p['text'])}")
        try:
            r = engine.generate(p["text"])
            print(f"    TPS: {r.tokens_per_sec} | Tokens: {r.tokens_generated} | "
                  f"Time: {r.total_sec}s | RAM: {r.ram_mb} MB")
            results.append({
                "label": p["label"],
                "tier": r.tier.value,
                "tokens_per_sec": r.tokens_per_sec,
                "tokens_generated": r.tokens_generated,
                "total_sec": r.total_sec,
                "ram_mb": r.ram_mb,
                "max_tokens_used": r.params.max_tokens,
            })
        except Exception as e:
            print(f"    INFERENCE FAILED: {e}")

    del model, tokenizer, engine
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        pass

    return {
        "model": name,
        "model_id": model_cfg["id"],
        "size_gb": model_cfg["size_gb"],
        "load_ms": load_ms,
        "results": results,
        "avg_tps": round(sum(r["tokens_per_sec"] for r in results) / max(len(results), 1), 1),
        "avg_ram": round(sum(r["ram_mb"] for r in results) / max(len(results), 1), 0),
    }


def write_report(all_results: list[dict]):
    out = Path("multi_model_results.json")
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nJSON saved: {out}")

    lines = [
        "# PRISM Multi-Model Benchmark",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Hardware:** Apple M1, {psutil.virtual_memory().total // 1024**3} GB RAM",
        f"**Platform:** {platform.platform()}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Model | Size | Load | avg TPS | avg RAM |",
        "|-------|------|------|---------|---------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['model']} | {r['size_gb']} GB | {r['load_ms']}ms | "
            f"**{r['avg_tps']}** | {r['avg_ram']} MB |"
        )

    lines += ["", "---", "", "## Per-Prompt Detail", ""]
    for r in all_results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| Prompt | Tier | TPS | Tokens | Time | RAM |")
        lines.append("|--------|------|-----|--------|------|-----|")
        for p in r["results"]:
            lines.append(
                f"| {p['label']} | {p['tier']} | {p['tokens_per_sec']} | "
                f"{p['tokens_generated']} | {p['total_sec']}s | {p['ram_mb']} MB |"
            )
        lines.append("")

    md = Path("MULTI_MODEL_RESULTS.md")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {md}")
    print("\n" + "\n".join(lines))


def main():
    print("PRISM Multi-Model Benchmark")
    print(f"RAM: {psutil.virtual_memory().total/1024**3:.1f} GB total, "
          f"{psutil.virtual_memory().available/1024**3:.1f} GB free")
    print()

    all_results = []
    for model_cfg in MODELS:
        result = benchmark_model(model_cfg)
        if result:
            all_results.append(result)

    if all_results:
        write_report(all_results)
    else:
        print("No models benchmarked - not enough RAM. Close other apps and retry.")


if __name__ == "__main__":
    main()
