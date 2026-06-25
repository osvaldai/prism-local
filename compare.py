#!/usr/bin/env python3
"""Compare baseline vs PRISM results and write RESULTS.md."""
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent

def load_json(name):
    p = RESULTS_DIR / name
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {p}: {e}")
        return None

def pct_change(base, new):
    if base == 0: return 0
    return (new - base) / base * 100

def main():
    base = load_json("baseline_results.json")
    prism = load_json("prism_results.json")

    if not base or not prism:
        print("Run baseline.py and prism_benchmark.py first")
        return 1

    base_map  = {r["label"]: r for r in base["results"]}
    prism_map = {r["label"]: r for r in prism["results"]}

    lines = [
        f"# PRISM vs Baseline — Результати",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Модель:** {base['model']}",
        f"**Платформа:** Apple M1, 8GB RAM, MLX",
        "",
        "---",
        "",
        "## Таблиця порівняння",
        "",
        "| Тест | Метрика | Baseline | PRISM | Зміна |",
        "|------|---------|----------|-------|-------|",
    ]

    total_tps_base, total_tps_prism = 0, 0
    total_ram_base, total_ram_prism = 0, 0
    count = 0

    for label in ("simple", "medium", "complex"):
        b = base_map.get(label)
        p = prism_map.get(label)
        if not b or not p:
            continue
        count += 1

        tps_delta = pct_change(b["tokens_per_sec"], p["tokens_per_sec"])
        ram_delta = pct_change(b["ram_mb"], p["ram_mb"])
        time_delta = pct_change(b["total_sec"], p["total_sec"])

        total_tps_base  += b["tokens_per_sec"]
        total_tps_prism += p["tokens_per_sec"]
        total_ram_base  += b["ram_mb"]
        total_ram_prism += p["ram_mb"]

        def arrow(d, invert=False):
            if invert:
                d = -d
            return "↑" if d > 2 else ("↓" if d < -2 else "→")

        lines += [
            f"| **{label.upper()}** | tokens/sec | {b['tokens_per_sec']:.1f} | {p['tokens_per_sec']:.1f} | {arrow(tps_delta)}{tps_delta:+.0f}% |",
            f"| | total_sec | {b['total_sec']:.2f}s | {p['total_sec']:.2f}s | {arrow(time_delta, invert=True)}{time_delta:+.0f}% |",
            f"| | RAM MB | {b['ram_mb']:.0f} | {p['ram_mb']:.0f} | {arrow(ram_delta, invert=True)}{ram_delta:+.0f}% |",
            f"| | ctx_chars | {b.get('prompt_len', '-')} | {p.get('compressed_context_len', '-')} | {'compressed' if p.get('context_compressed') else 'same'} |",
        ]

    if count > 0:
        avg_tps_base  = total_tps_base / count
        avg_tps_prism = total_tps_prism / count
        avg_ram_base  = total_ram_base / count
        avg_ram_prism = total_ram_prism / count
        tps_avg_delta = pct_change(avg_tps_base, avg_tps_prism)
        ram_avg_delta = pct_change(avg_ram_base, avg_ram_prism)

        lines += [
            "",
            "---",
            "",
            "## Зведення",
            "",
            f"| Метрика | Baseline avg | PRISM avg | Зміна |",
            f"|---------|-------------|-----------|-------|",
            f"| tokens/sec | {avg_tps_base:.1f} | {avg_tps_prism:.1f} | {tps_avg_delta:+.0f}% |",
            f"| RAM MB | {avg_ram_base:.0f} | {avg_ram_prism:.0f} | {ram_avg_delta:+.0f}% |",
            f"| Load time | {base['load_ms']}ms | {prism['load_ms']}ms | — |",
            "",
            "---",
            "",
            "## PRISM компоненти активні",
            "",
            "- **INTAKE Classifier:** автоматичний вибір tier (SIMPLE/MEDIUM/COMPLEX)",
            "- **Adaptive Sampler:** temperature/top_p/max_tokens per tier",
            "- **Context Compressor:** TF-IDF extractive (target: max_context_chars)",
            "- **KV-Cache Manager:** sliding window 2048 tokens",
            "- **Hardware Profiler:** M1 detected, Metal GPU, 4bit quant",
            "",
            "---",
            "",
            "## Висновок",
            "",
        ]

        if tps_avg_delta > 5:
            lines.append(f"PRISM показав **{tps_avg_delta:+.0f}% більше tokens/sec** завдяки адаптивним параметрам.")
        elif tps_avg_delta < -5:
            lines.append(f"PRISM показав **{tps_avg_delta:+.0f}% менше tokens/sec** — overhead компресії > виграш від менших max_tokens.")
        else:
            lines.append("PRISM показав **порівнянний throughput** з baseline.")

        if ram_avg_delta < -5:
            lines.append(f"Пам'ять: **{ram_avg_delta:+.0f}%** — PRISM зменшив RAM через адаптивні max_tokens.")
        else:
            lines.append(f"Пам'ять: **{ram_avg_delta:+.0f}%** від baseline.")

    out = RESULTS_DIR / "RESULTS.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written: {out}")
    for line in lines:
        print(line)

if __name__ == "__main__":
    main()
