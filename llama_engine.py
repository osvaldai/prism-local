#!/usr/bin/env python3
"""
PRISM Llama.cpp Engine — 70B model support via llama-cpp-python.

Enables running GGUF models (3B–70B) locally with:
  - Metal GPU partial offload (Apple Silicon unified memory)
  - mmap: model lives on disk, pages demand-loaded (handles 40GB+ on 8GB RAM)
  - PRISM algorithm: same tier/compress/budget logic as prism_engine_v2.py
  - Streaming token output

Performance on M1 8GB (realistic expectations with mmap):
  7B  Q4 (3.8GB)  → 8–15 TPS  (fits in RAM)
  13B Q4 (7.2GB)  → 3–6 TPS   (partial swap)
  70B Q2 (19GB)   → 0.5–2 TPS (heavy swap, 5-10 GPU layers)
  70B Q4 (40GB)   → 0.3–1 TPS (full swap, only useful for quality)
"""
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional
import psutil

from prism_engine_v2 import (
    Tier, SamplingParams, PRISMResult, HardwareProfile,
    classify_complexity, compress_context, _dynamic_budget, profile_hardware,
)


# ─── Known GGUF Models ────────────────────────────────────────────────────────

KNOWN_MODELS = {
    # Small — fit in 8GB RAM
    "gemma-3-4b-q4":    "bartowski/gemma-3-4b-it-GGUF / gemma-3-4b-it-Q4_K_M.gguf",
    "llama-3.2-3b-q4":  "bartowski/Llama-3.2-3B-Instruct-GGUF / Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "qwen2.5-7b-q4":    "bartowski/Qwen2.5-7B-Instruct-GGUF / Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    # Medium — 16GB recommended
    "llama-3.1-8b-q4":  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF / Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "mistral-7b-q4":    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF / mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "gemma-3-12b-q4":   "bartowski/gemma-3-12b-it-GGUF / gemma-3-12b-it-Q4_K_M.gguf",
    # Large — 32GB+ recommended (mmap allows 8GB but slow)
    "llama-3.1-70b-q4": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF / Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    "llama-3.1-70b-q2": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF / Meta-Llama-3.1-70B-Instruct-Q2_K.gguf",
    "qwen2.5-72b-q2":   "bartowski/Qwen2.5-72B-Instruct-GGUF / Qwen2.5-72B-Instruct-Q2_K_L.gguf",
}


def list_known_models() -> None:
    print("\nKnown GGUF models (download from HuggingFace):")
    print(f"  {'Key':<22} HF repo / filename")
    print("  " + "-" * 70)
    for key, info in KNOWN_MODELS.items():
        print(f"  {key:<22} {info}")
    print()
    print("  Download example:")
    print("  huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \\")
    print("    Meta-Llama-3.1-70B-Instruct-Q2_K.gguf --local-dir ./models/llama-70b/")


# ─── Auto GPU layers ─────────────────────────────────────────────────────────

def _auto_gpu_layers(model_path: str, hw: HardwareProfile) -> int:
    """
    Estimate n_gpu_layers for Metal offload on Apple Silicon.
    Unified memory: GPU layers physically stay in system RAM.
    Reserve 1.5GB for OS + KV cache activations.
    """
    if not hw.has_metal_gpu:
        return 0

    p = Path(model_path)
    model_gb = p.stat().st_size / 1024**3 if p.exists() else 0.0
    free_for_gpu = max(0.0, hw.free_ram_gb - 1.5)

    # Estimate layer count and per-layer size
    if model_gb < 5:
        n_layers, layer_gb = 32, model_gb / 32
    elif model_gb < 10:
        n_layers, layer_gb = 40, model_gb / 40
    elif model_gb < 20:
        n_layers, layer_gb = 60, model_gb / 60
    else:
        n_layers, layer_gb = 80, model_gb / 80

    if layer_gb <= 0:
        return 0

    n_gpu = int(free_for_gpu / layer_gb)
    return min(n_gpu, n_layers)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class LlamaConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = -1          # -1 = auto-detect
    n_threads: Optional[int] = None # None = cpu_cores
    use_mmap: bool = True           # model pages from disk (critical for 70B on 8GB)
    use_mlock: bool = False         # pin loaded pages in RAM (use if enough free RAM)
    verbose: bool = False


# ─── Engine ──────────────────────────────────────────────────────────────────

class LlamaEngine:
    """
    PRISM-wrapped llama.cpp inference engine.
    Same generate() / generate_stream() API as PRISMEngineV2.
    """

    def __init__(self, model_path: str, config: Optional[LlamaConfig] = None):
        self.model_path = model_path
        self.config = config or LlamaConfig(model_path=model_path)
        self.hw = profile_hardware()
        self._llm = None
        self._load()

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed.\n"
                "Run: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
            )

        cfg = self.config
        n_gpu = cfg.n_gpu_layers
        if n_gpu == -1:
            n_gpu = _auto_gpu_layers(cfg.model_path, self.hw)
            print(f"  Auto n_gpu_layers={n_gpu}")

        n_threads = cfg.n_threads or self.hw.cpu_cores

        p = Path(cfg.model_path)
        model_gb = p.stat().st_size / 1024**3 if p.exists() else 0.0
        print(
            f"  {p.name} | {model_gb:.1f}GB | "
            f"RAM free={self.hw.free_ram_gb:.1f}GB | "
            f"gpu_layers={n_gpu} | threads={n_threads} | mmap={cfg.use_mmap}"
        )
        if model_gb > self.hw.free_ram_gb * 1.2:
            print(
                f"  NOTE: model > free RAM — NVMe swap active, "
                "first tokens will be slow (page faults). Normal after warm-up."
            )

        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=n_gpu,
            n_threads=n_threads,
            use_mmap=cfg.use_mmap,
            use_mlock=cfg.use_mlock,
            verbose=cfg.verbose,
        )
        print(f"  Loaded in {int((time.perf_counter()-t0)*1000)}ms")

    # ── Blocking generate ─────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: str = "") -> PRISMResult:
        parts: list[str] = []
        final: Optional[PRISMResult] = None
        for tok, meta in self.generate_stream(prompt, system_prompt):
            if meta is None:
                parts.append(tok)
            else:
                final = meta
        if final is not None:
            final.output = "".join(parts)
            return final
        return PRISMResult(
            output="".join(parts), tier=Tier.SIMPLE,
            tokens_generated=0, total_sec=0, tokens_per_sec=0,
            ram_mb=0, context_compressed=False,
            original_context_len=0, compressed_context_len=0,
            params=SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95),
        )

    # ── Streaming generate ────────────────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        history: "list[dict] | None" = None,
    ) -> Generator[tuple[str, Optional[PRISMResult]], None, None]:
        tier = classify_complexity(prompt)
        live_free_ram = psutil.virtual_memory().available / 1024**3
        params = _dynamic_budget(prompt, tier, free_ram_gb=live_free_ram)

        original_len = len(prompt)
        ctx_limit = self.config.n_ctx * 3
        compressed_prompt = compress_context(prompt, ctx_limit // 2, query=prompt)
        compressed_len = len(compressed_prompt)

        # Build history-aware formatted prompt
        formatted = self._format_prompt_with_history(compressed_prompt, system_prompt, history)

        proc = psutil.Process()
        peak_rss = proc.memory_info().rss
        gc.collect()
        t0 = time.perf_counter()
        ttft = 0.0
        parts: list[str] = []
        first_tok = True

        stop_tokens = ["<|eot_id|>", "<end_of_turn>", "[/INST]", "</s>",
                       "<|im_end|>", "<|endoftext|>"]
        _rep_buf = ""
        _total_len = 0
        try:
            for chunk in self._llm(
                formatted,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                repeat_penalty=1.1,
                stream=True,
                stop=stop_tokens,
            ):
                tok = chunk["choices"][0]["text"]
                if first_tok and tok:
                    ttft = time.perf_counter() - t0
                    first_tok = False
                cur_rss = proc.memory_info().rss
                if cur_rss > peak_rss:
                    peak_rss = cur_rss
                parts.append(tok)
                yield tok, None
                # Repetition early-stop
                _total_len += len(tok)
                if _total_len > 200:
                    _rep_buf += tok
                    if len(_rep_buf) > 180:
                        _rep_buf = _rep_buf[-180:]
                    tail = _rep_buf[-60:]
                    if len(tail) == 60 and _rep_buf[:-60].count(tail) >= 2:
                        break
        except Exception as e:
            yield f"\n[LlamaEngine error: {e}]", None

        output = "".join(parts)
        elapsed = time.perf_counter() - t0
        ram_mb = proc.memory_info().rss / 1024**2
        n_tokens = max(len(output.split()) * 4 // 3, 1)
        tps = n_tokens / elapsed if elapsed > 0 else 0

        yield "", PRISMResult(
            output=output,
            tier=tier,
            tokens_generated=n_tokens,
            total_sec=round(elapsed, 3),
            tokens_per_sec=round(tps, 1),
            ram_mb=round(ram_mb, 0),
            peak_ram_mb=round(peak_rss / 1024**2, 0),
            ttft_sec=round(ttft, 3),
            context_compressed=(compressed_len < original_len),
            original_context_len=original_len,
            compressed_context_len=compressed_len,
            params=params,
        )

    def _format_prompt_with_history(
        self, text: str, system_prompt: str = "", history: "list[dict] | None" = None
    ) -> str:
        name = Path(self.model_path).name.lower()
        sys = system_prompt or "You are a helpful assistant."
        turns = history or []

        if "llama-3" in name or "llama3" in name:
            parts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}<|eot_id|>"]
            for t in turns:
                role = "user" if t["role"] == "user" else "assistant"
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{t['content']}<|eot_id|>")
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
            return "".join(parts)

        if "gemma" in name:
            parts = [f"<start_of_turn>system\n{sys}<end_of_turn>\n"]
            for t in turns:
                role = "user" if t["role"] == "user" else "model"
                parts.append(f"<start_of_turn>{role}\n{t['content']}<end_of_turn>\n")
            parts.append(f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n")
            return "".join(parts)

        if "mistral" in name or "mixtral" in name:
            conv = "".join(
                f"[INST] {t['content']} [/INST] " if t["role"] == "user"
                else f"{t['content']} "
                for t in turns
            )
            return f"[INST] {sys}\n\n{conv}{text} [/INST]"

        if "qwen" in name:
            parts = [f"<|im_start|>system\n{sys}<|im_end|>\n"]
            for t in turns:
                parts.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n")
            parts.append(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
            return "".join(parts)

        conv = "".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n" for t in turns
        )
        return f"System: {sys}\n\n{conv}User: {text}\nAssistant:"

    def _format_prompt(self, text: str, system_prompt: str = "") -> str:
        name = Path(self.model_path).name.lower()
        sys = system_prompt or "You are a helpful assistant."

        if "llama-3" in name or "llama3" in name:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{sys}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
        if "gemma" in name:
            return (
                f"<start_of_turn>system\n{sys}<end_of_turn>\n"
                f"<start_of_turn>user\n{text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        if "mistral" in name or "mixtral" in name:
            return f"[INST] {sys}\n\n{text} [/INST]"
        if "qwen" in name:
            return (
                f"<|im_start|>system\n{sys}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        # Generic fallback
        return f"System: {sys}\n\nUser: {text}\nAssistant:"

    def print_profile(self):
        print(
            f"llama.cpp | {Path(self.model_path).name} | "
            f"ctx={self.config.n_ctx} | mmap={self.config.use_mmap} | "
            f"RAM free={self.hw.free_ram_gb:.1f}GB"
        )
