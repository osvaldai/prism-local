#!/usr/bin/env python3
"""
PRISM v2 Backtest — component verification + optional full inference.

Tests:
  1. Classifier accuracy    (20 labelled prompts)
  2. BM25 vs TF-IDF quality (relevance score on held-out sentence)
  3. Dynamic budget          (per tier + RAM-aware cap)
  4. Code block preservation (no mid-block splits)
  5. Full inference          (if --model provided, writes BACKTEST_V2.md)

Run:
    python backtest_v2.py
    python backtest_v2.py --model mlx-community/gemma-3-4b-it-4bit
"""
import argparse
import re
import time
from collections import Counter

from prism_engine_v2 import (
    Tier, classify_complexity, compress_context,
    _dynamic_budget, _split_preserving_code, profile_hardware,
)

PASS = "PASS"
FAIL = "FAIL"


# ─── v1 TF-IDF for comparison ─────────────────────────────────────────────────

def _tfidf_compress(text: str, target_chars: int) -> str:
    def tok(t):
        return [w for w in re.findall(r"[\w']+", t.lower()) if len(w) > 2]
    if len(text) <= target_chars:
        return text
    sents = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if len(s.strip()) > 10]
    if len(sents) <= 3:
        return text[:target_chars]
    tf: Counter = Counter()
    for s in sents:
        tf.update(tok(s))
    max_tf = max(tf.values(), default=1)
    scored = [(sum(tf[w] for w in tok(s)) / max(len(tok(s)), 1) / max_tf, i, s)
              for i, s in enumerate(sents)]
    scored.sort(reverse=True)
    budget, selected = target_chars, []
    for _, idx, s in scored:
        if budget <= 0: break
        if len(s) <= budget:
            selected.append((idx, s)); budget -= len(s) + 1
    selected.sort()
    return " ".join(s for _, s in selected) or text[:target_chars]


# ─── 1. Classifier ───────────────────────────────────────────────────────────

CASES = [
    ("What is 2+2?",                        Tier.SIMPLE),
    ("Hi",                                   Tier.SIMPLE),
    ("What time is it?",                     Tier.SIMPLE),
    ("Who invented the telephone?",          Tier.SIMPLE),
    ("Define recursion.",                    Tier.SIMPLE),
    ("What is Python?",                      Tier.SIMPLE),
    ("Explain how neural networks learn.",   Tier.MEDIUM),
    ("Differences between TCP and UDP?",     Tier.MEDIUM),
    ("How does garbage collection work?",    Tier.MEDIUM),
    ("Summarize causes of World War I.",     Tier.MEDIUM),
    ("Write a function to reverse a string.", Tier.MEDIUM),
    (
        "Analyze tradeoffs between transformer attention and RNNs for long-sequence "
        "modeling, considering memory complexity, parallelism, and edge deployment.",
        Tier.COMPLEX,
    ),
    (
        "Compare Redis, Memcached, and Aerospike for high-throughput distributed caching. "
        "Include consistency models, eviction policies, and cluster management.",
        Tier.COMPLEX,
    ),
    (
        "Design a fault-tolerant distributed key-value store with Paxos consensus, "
        "linearizable reads, and automatic leader election. Discuss CAP implications.",
        Tier.COMPLEX,
    ),
    (
        "```python\ndef merge_sort(arr):\n    pass\n```\n"
        "Implement merge sort, explain complexity, compare with quicksort.",
        Tier.COMPLEX,
    ),
    (
        "Should we migrate monolith to microservices? First consider team size, "
        "then deployment complexity, then security implications of service mesh.",
        Tier.COMPLEX,
    ),
]


def test_classifier() -> float:
    print("\n=== 1. Classifier Accuracy ===")
    ok_count = 0
    for prompt, expected in CASES:
        got = classify_complexity(prompt)
        ok = got == expected
        ok_count += ok
        tag = PASS if ok else FAIL
        print(f"  [{tag}] exp={expected.value:<8} got={got.value:<8} | {prompt[:58]}…")
    acc = ok_count / len(CASES) * 100
    print(f"\n  Accuracy: {ok_count}/{len(CASES)} = {acc:.0f}%")
    return acc


# ─── 2. Compression quality ───────────────────────────────────────────────────

_CTX = """
The sky is blue during the day.
Python uses reference counting combined with a cyclic garbage collector.
Cats are popular household pets.
The garbage collector in Python uses cyclic reference detection to free memory.
Mount Everest is the tallest mountain on Earth.
Cyclic references in Python are handled by the gc module using generational collection.
The Amazon river flows through South America.
Python memory management relies on reference counting as the primary mechanism.
The Eiffel Tower is located in Paris.
Memory leaks in Python often occur due to uncollected cyclic reference chains.
"""
_QUERY = "How does Python garbage collector handle cyclic references?"
_KEYWORDS = {"python", "garbage", "collector", "cyclic", "reference", "memory"}
_TARGET = 200


def _relevance(text: str) -> float:
    """F1-like: penalises irrelevant sentences, rewards keyword density."""
    words = re.findall(r"[\w]+", text.lower())
    word_set = set(words)
    recall    = len(word_set & _KEYWORDS) / len(_KEYWORDS)
    precision = sum(1 for w in words if w in _KEYWORDS) / max(len(words), 1)
    return round((recall + precision) / 2, 3)


def test_compression() -> tuple[float, float]:
    print("\n=== 2. BM25 v2 vs TF-IDF v1 Compression ===")

    t0 = time.perf_counter()
    bm25_out = compress_context(_CTX.strip(), _TARGET, query=_QUERY)
    bm25_ms  = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tfidf_out = _tfidf_compress(_CTX.strip(), _TARGET)
    tfidf_ms  = (time.perf_counter() - t0) * 1000

    br, tr = _relevance(bm25_out), _relevance(tfidf_out)
    print(f"  BM25:   relevance={br:.2f}  len={len(bm25_out)}ch  {bm25_ms:.2f}ms")
    print(f"  TF-IDF: relevance={tr:.2f}  len={len(tfidf_out)}ch  {tfidf_ms:.2f}ms")
    print(f"  BM25 out:   {bm25_out[:100]}…")
    print(f"  TF-IDF out: {tfidf_out[:100]}…")
    winner = "BM25 v2" if br >= tr else "TF-IDF v1"
    print(f"  Winner: {winner}  (Δ={br-tr:+.2f})")
    return br, tr


# ─── 3. Dynamic budget ────────────────────────────────────────────────────────

BUDGET_CASES = [
    # prompt                                tier          expected  free_ram_gb
    ("Hi",                                 Tier.SIMPLE,  48,   99.0),  # n<30 → 48
    ("What is 2+2? Short answer please.",  Tier.SIMPLE,  96,   99.0),  # n>30 → 96
    ("Explain X.",                         Tier.MEDIUM,  512,  99.0),
    ("Write ```python def f(): pass```.",  Tier.MEDIUM,  768,  99.0),  # code syntax → 768
    ("Compare A vs B and list tradeoffs.", Tier.COMPLEX, 2048, 99.0),
    ("Explain X.",                         Tier.MEDIUM,  256,  1.0),   # low RAM cap
    ("Deep analysis of systems.",          Tier.COMPLEX, 256,  1.0),   # low RAM cap (< 1.5GB → 256)
]


def test_budget() -> float:
    print("\n=== 3. Dynamic Token Budget ===")
    ok_count = 0
    for prompt, tier, expected, free_ram in BUDGET_CASES:
        p = _dynamic_budget(prompt, tier, free_ram_gb=free_ram)
        ok = p.max_tokens == expected
        ok_count += ok
        tag = PASS if ok else FAIL
        print(f"  [{tag}] tier={tier.value:<8} ram={free_ram:.0f}GB "
              f"exp={expected:<5} got={p.max_tokens:<5} | {prompt[:40]}")
    acc = ok_count / len(BUDGET_CASES)
    print(f"\n  Budget accuracy: {ok_count}/{len(BUDGET_CASES)}")
    return acc


# ─── 4. Code block preservation ──────────────────────────────────────────────

_CODE_TEXT = """
First sentence about Python.
Here is a code example:
```python
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
```
Last sentence after the code block.
"""


def test_code_preservation() -> bool:
    print("\n=== 4. Code Block Preservation ===")
    segs = _split_preserving_code(_CODE_TEXT)
    blocks = [s for s in segs if s.startswith("```")]
    ok = len(blocks) == 1 and "def add" in blocks[0] and "return" in blocks[0]
    tag = PASS if ok else FAIL
    print(f"  [{tag}] Code blocks kept atomic: {len(blocks)}")
    if blocks:
        print(f"         Block: {blocks[0][:100]}…")
    return ok


# ─── 5. Full inference ────────────────────────────────────────────────────────

INFERENCE_PROMPTS = [
    {"label": "simple",  "text": "What is 2+2? Short answer."},
    {"label": "medium",  "text": "Explain the attention mechanism in transformers briefly."},
    {"label": "complex", "text": (
        "Compare sliding window attention vs full self-attention for long-context LLM "
        "inference. Include memory complexity, throughput, and edge device constraints."
    )},
]


def test_inference(model_id: str):
    print(f"\n=== 5. Full Inference: {model_id} ===")
    from prism_engine_v2 import load_prism

    print("  Loading…")
    t0 = time.perf_counter()
    engine = load_prism(model_id)
    load_ms = int((time.perf_counter() - t0) * 1000)
    print(f"  Load time: {load_ms}ms")
    engine.print_profile()

    results = []
    for p in INFERENCE_PROMPTS:
        tier = classify_complexity(p["text"])
        print(f"\n  [{p['label'].upper()}] tier={tier.value}  len={len(p['text'])}")
        r = engine.generate(p["text"])
        print(f"    TPS={r.tokens_per_sec} tokens={r.tokens_generated} "
              f"time={r.total_sec}s RAM={r.ram_mb:.0f}MB "
              f"compress={r.context_compressed} spec={r.speculative_used}")
        results.append({
            "label": p["label"], "tier": r.tier.value,
            "tps": r.tokens_per_sec, "tokens": r.tokens_generated,
            "time_s": r.total_sec, "ram_mb": r.ram_mb,
        })

    avg_tps = sum(r["tps"] for r in results) / len(results)
    print(f"\n  avg TPS: {avg_tps:.1f}")
    _write_results(results, model_id, avg_tps)
    return results


def _write_results(results, model_id, avg_tps):
    import json, time as t
    import platform

    data = {
        "date": t.strftime("%Y-%m-%d %H:%M"),
        "model": model_id,
        "avg_tps": round(avg_tps, 1),
        "hw": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "results": results,
    }
    with open("backtest_v2_results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    lines = [
        "# PRISM v2 Backtest Results",
        "",
        f"**Date:** {data['date']}  **Model:** {model_id}  **avg TPS:** {avg_tps:.1f}",
        "",
        "| Label | Tier | TPS | Tokens | Time | RAM |",
        "|-------|------|-----|--------|------|-----|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['tier']} | {r['tps']} | "
            f"{r['tokens']} | {r['time_s']}s | {r['ram_mb']:.0f}MB |"
        )
    with open("BACKTEST_V2.md", "w") as f:
        f.write("\n".join(lines))
    print("  Wrote: backtest_v2_results.json  BACKTEST_V2.md")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PRISM v2 Backtest")
    ap.add_argument("--model", default="", help="MLX model ID for inference test (optional)")
    args = ap.parse_args()

    print("=" * 60)
    print("PRISM v2 Backtest")
    print("=" * 60)

    hw = profile_hardware()
    print(f"\nHW: {hw.total_ram_gb}GB total | {hw.free_ram_gb:.1f}GB free | "
          f"{hw.cpu_cores} cores | Metal={'yes' if hw.has_metal_gpu else 'no'}")

    acc     = test_classifier()
    br, tr  = test_compression()
    bacc    = test_budget()
    code_ok = test_code_preservation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Classifier accuracy:  {acc:.0f}%")
    print(f"  BM25 relevance:       {br:.2f}  (TF-IDF: {tr:.2f}  Δ={br-tr:+.2f})")
    print(f"  Budget accuracy:      {bacc*100:.0f}%")
    print(f"  Code preservation:    {PASS if code_ok else FAIL}")

    if args.model:
        test_inference(args.model)
    else:
        print("\n  Add --model mlx-community/gemma-3-4b-it-4bit for full inference test")


if __name__ == "__main__":
    main()
