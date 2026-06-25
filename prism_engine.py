#!/usr/bin/env python3
"""
PRISM Engine — Progressive Resolution Inference with Selective Magnification
Universal AI inference optimization wrapper for local MLX models.

Components:
  1. INTAKE Classifier     — complexity scoring, no neural net
  2. Adaptive Sampler      — temperature/top_p/max_tokens per complexity tier
  3. KV-Cache Manager      — sliding window to prevent OOM on long contexts
  4. Context Compressor    — TF-IDF extractive compression
  5. Hardware Profiler     — RAM/CPU/GPU detection, quantization tier
"""
import gc
import re
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import psutil


# ─── Complexity Tiers ────────────────────────────────────────────────────────

class Tier(Enum):
    SIMPLE  = "simple"
    MEDIUM  = "medium"
    COMPLEX = "complex"


@dataclass
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: Optional[float] = 1.1


TIER_PARAMS = {
    Tier.SIMPLE:  SamplingParams(max_tokens=128,  temperature=0.1, top_p=0.90),
    Tier.MEDIUM:  SamplingParams(max_tokens=512,  temperature=0.7, top_p=0.95),
    Tier.COMPLEX: SamplingParams(max_tokens=1024, temperature=1.0, top_p=1.00),
}


# ─── 1. INTAKE Classifier ────────────────────────────────────────────────────

DECISION_RE = re.compile(r"\b(виріш|домов|decided|agreed|conclusion|verdict)\b", re.I)
ACTION_RE   = re.compile(r"\b(треба|повинен|must|should|will|need to|have to)\b", re.I)
QUESTION_RE = re.compile(r"\?|питання|question|why|how|when|who|what", re.I)
NEGATE_RE   = re.compile(r"\b(not|no|never|isn't|aren't|wasn't|doesn't|don't)\b", re.I)


def classify_complexity(text: str) -> Tier:
    # Short factual questions are always SIMPLE regardless of unique_ratio
    if len(text) < 50 and not DECISION_RE.search(text) and not ACTION_RE.search(text):
        return Tier.SIMPLE

    words = re.findall(r"[\w']+", text.lower())
    unique_ratio = len(set(words)) / max(len(words), 1)
    length_score = min(len(text) / 300, 0.6)
    struct_score = min(
        (bool(QUESTION_RE.search(text)) * 0.1 +
         bool(NEGATE_RE.search(text)) * 0.1 +
         bool(DECISION_RE.search(text)) * 0.1 +
         text.count(",") * 0.01 +
         text.count(";") * 0.02),
        0.4,
    )
    score = length_score + unique_ratio * 0.15 + struct_score
    if score < 0.25:
        return Tier.SIMPLE
    if score < 0.55:
        return Tier.MEDIUM
    return Tier.COMPLEX


# ─── 2. Context Compressor (TF-IDF extractive) ───────────────────────────────

def _word_tokens(text: str):
    return [w for w in re.findall(r"[\w']+", text.lower()) if len(w) > 2]


def compress_context(text: str, target_chars: int) -> str:
    """TF-IDF extractive sentence selection — PRISM Component 4."""
    if len(text) <= target_chars:
        return text

    sents = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if len(s.strip()) > 10]
    if len(sents) <= 3:
        return text[:target_chars]

    tf: Counter = Counter()
    for s in sents:
        tf.update(_word_tokens(s))
    max_tf = max(tf.values(), default=1)

    scored = []
    for i, s in enumerate(sents):
        words = _word_tokens(s)
        score = sum(tf[w] for w in words) / max(len(words), 1) / max_tf
        if DECISION_RE.search(s): score *= 1.5
        if ACTION_RE.search(s):   score *= 1.4
        if "?" in s:               score *= 1.3
        scored.append((score, i, s))

    scored.sort(reverse=True)
    budget = target_chars
    selected = []
    for score, idx, s in scored:
        if budget <= 0:
            break
        if len(s) <= budget:
            selected.append((idx, s))
            budget -= len(s) + 1

    selected.sort()
    return " ".join(s for _, s in selected) or text[:target_chars]


# ─── 3. KV-Cache Sliding Window Manager ──────────────────────────────────────

class KVCacheManager:
    """Limits context to prevent OOM. Keeps anchor (first N) + recent (last M)."""

    def __init__(self, max_tokens: int = 2048, anchor_tokens: int = 64):
        self.max_tokens = max_tokens
        self.anchor_tokens = anchor_tokens
        self._history: list[str] = []
        self._token_count = 0

    def add(self, text: str, approx_tokens_per_char: float = 0.25):
        self._history.append(text)
        self._token_count += int(len(text) * approx_tokens_per_char)
        self._trim()

    def get_context(self) -> str:
        return "\n".join(self._history)

    def _trim(self):
        while self._token_count > self.max_tokens and len(self._history) > 2:
            removed = self._history.pop(1)  # keep first (system), remove oldest middle
            self._token_count -= int(len(removed) * 0.25)

    def reset(self):
        self._history.clear()
        self._token_count = 0


# ─── 4. Hardware Profiler ─────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    total_ram_gb: float
    free_ram_gb: float
    cpu_cores: int
    has_metal_gpu: bool
    recommended_quant: str      # "4bit" | "8bit" | "16bit"
    max_context_chars: int


def profile_hardware() -> HardwareProfile:
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024**3
    free_gb  = mem.available / 1024**3
    cores    = psutil.cpu_count(logical=False) or 1

    import platform
    has_metal = platform.system() == "Darwin" and platform.machine() in ("arm64", "x86_64")

    if free_gb < 3:
        quant = "4bit"
        ctx = 4_000
    elif free_gb < 6:
        quant = "4bit"
        ctx = 8_000
    elif free_gb < 12:
        quant = "8bit"
        ctx = 16_000
    else:
        quant = "16bit"
        ctx = 32_000

    return HardwareProfile(
        total_ram_gb=round(total_gb, 1),
        free_ram_gb=round(free_gb, 1),
        cpu_cores=cores,
        has_metal_gpu=has_metal,
        recommended_quant=quant,
        max_context_chars=ctx,
    )


# ─── 5. PRISM Engine ─────────────────────────────────────────────────────────

@dataclass
class PRISMResult:
    output: str
    tier: Tier
    tokens_generated: int
    total_sec: float
    tokens_per_sec: float
    ram_mb: float
    context_compressed: bool
    original_context_len: int
    compressed_context_len: int
    params: SamplingParams


class PRISMEngine:
    """PRISM inference wrapper. Wraps any MLX model with adaptive optimization."""

    def __init__(self, model, tokenizer, hw: Optional[HardwareProfile] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.hw = hw or profile_hardware()
        self.kv_cache = KVCacheManager(max_tokens=2048, anchor_tokens=64)

    def generate(self, prompt: str, system_prompt: str = "") -> PRISMResult:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        # 1. Classify complexity
        tier = classify_complexity(prompt)
        params = TIER_PARAMS[tier]

        # 2. Context compression
        full_context = (system_prompt + "\n" + prompt).strip() if system_prompt else prompt
        original_len = len(full_context)
        compressed = compress_context(full_context, self.hw.max_context_chars)
        compressed_len = len(compressed)

        # 3. Build sampler with adaptive temperature/top_p (mlx_lm 0.31+ API)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)

        proc = psutil.Process()
        gc.collect()

        t0 = time.perf_counter()
        output = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=compressed,
            max_tokens=params.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        ram_peak = proc.memory_info().rss / 1024**2
        out_tokens = len(self.tokenizer.encode(output)) if hasattr(self.tokenizer, "encode") else len(output.split())
        tps = out_tokens / elapsed if elapsed > 0 else 0

        return PRISMResult(
            output=output,
            tier=tier,
            tokens_generated=out_tokens,
            total_sec=round(elapsed, 3),
            tokens_per_sec=round(tps, 1),
            ram_mb=round(ram_peak, 0),
            context_compressed=(compressed_len < original_len),
            original_context_len=original_len,
            compressed_context_len=compressed_len,
            params=params,
        )

    def print_profile(self):
        hw = self.hw
        print(f"Hardware: {hw.total_ram_gb}GB RAM | {hw.free_ram_gb}GB free | "
              f"{hw.cpu_cores} cores | Metal={'yes' if hw.has_metal_gpu else 'no'}")
        print(f"Config: quant={hw.recommended_quant} | max_ctx={hw.max_context_chars} chars")
