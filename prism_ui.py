#!/usr/bin/env python3
"""
PRISM UI v3 — Terminal chat + live system monitoring.

Fixes vs v2:
  - MetricsWidget.update() renamed to update_metrics() (shadowed Widget.update)
  - self._busy set via call_from_thread (was mutated in background thread)
  - SystemMonitor widget: CPU%, RAM, memory pressure, chip name, TPS history sparkline

Run:
    python prism_ui.py
    python prism_ui.py --model mlx-community/gemma-3-4b-it-4bit
    python prism_ui.py --gguf ./models/llama-70b-q2.gguf

Install:
    bash setup_v2.sh
"""
import argparse
import platform
import subprocess
import threading
_engine_lock = threading.Lock()
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, RichLog, Select, Static, TextArea
from rich.text import Text

from prism_engine_v2 import PRISMResult, Tier, profile_hardware


# ─── Presets ─────────────────────────────────────────────────────────────────

MLX_PRESETS = [
    ("Gemma 3 4B INT4  — 3.3 GB (recommended)",  "mlx-community/gemma-3-4b-it-4bit"),
    ("Gemma 3 12B INT4 — 6.5 GB (16GB RAM)",      "mlx-community/gemma-3-12b-it-4bit"),
    ("Llama 3.1 8B INT4 — 4.7 GB",               "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
    ("Mistral 7B INT4  — 4.1 GB",                "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    ("Qwen 2.5 7B INT4 — 4.3 GB",               "mlx-community/Qwen2.5-7B-Instruct-4bit"),
]

DEFAULT_SYSTEM = "You are a helpful, concise assistant."

# TPS sparkline chars (ascending)
_SPARK = "▁▂▃▄▅▆▇█"


# ─── App state ───────────────────────────────────────────────────────────────

_HISTORY_MAX_TURNS = 40  # cap at 20 user+assistant pairs to prevent memory growth

@dataclass
class _State:
    engine: object = None
    engine_name: str = ""
    loading: bool = False
    last_result: Optional[PRISMResult] = None
    history: list[dict] = field(default_factory=list)
    tps_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def append_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > _HISTORY_MAX_TURNS:
            # Drop oldest pair (user+assistant) keeping system structure intact
            self.history = self.history[-_HISTORY_MAX_TURNS:]


_S = _State()


# ─── Chip detection ──────────────────────────────────────────────────────────

def _chip_name() -> str:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=1,
        )
        name = r.stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return platform.machine()


_CHIP = _chip_name()


# ─── Background loader ────────────────────────────────────────────────────────

def _load_thread(mode: str, model_id: str, gguf_path: str, draft_id: str, on_done, on_err):
    try:
        if mode == "mlx":
            from prism_engine_v2 import load_prism
            with _engine_lock:
                _S.engine = load_prism(model_id, draft_model_id=draft_id or None)
                _S.engine_name = model_id.split("/")[-1]
        else:
            from llama_engine import LlamaEngine
            with _engine_lock:
                _S.engine = LlamaEngine(gguf_path)
                _S.engine_name = Path(gguf_path).name
        _S.loading = False
        on_done()
    except Exception as e:
        _S.loading = False
        on_err(str(e))


# ─── CSS ─────────────────────────────────────────────────────────────────────

APP_CSS = """
Screen { background: #0d1117; }
Header { background: #161b22; color: #e6edf3; height: 3; }
Footer { background: #161b22; color: #8b949e; }

#layout { height: 1fr; }

/* ── Sidebar (Ctrl+B to toggle) ── */
#sidebar {
    width: 20;
    min-width: 20;
    background: #161b22;
    border-right: solid #30363d;
    padding: 1 1;
}
#sidebar.hidden { display: none; }

/* ── Chat takes all remaining space ── */
#chat-panel { width: 1fr; background: #0d1117; }

/* ── Right panel (Ctrl+M to toggle) ── */
#right-panel {
    width: 22;
    min-width: 22;
    background: #161b22;
    border-left: solid #30363d;
}
#right-panel.hidden { display: none; }

#metrics-panel {
    height: auto;
    padding: 1 1;
    border-bottom: solid #30363d;
}

#sysmon-panel {
    height: 1fr;
    padding: 1 1;
}

/* ── Chat log: no border, full height, dark bg ── */
#chat-log {
    height: 1fr;
    background: #010409;
    padding: 0 2;
    border: none;
}

/* ── Compact input row ── */
#input-row {
    height: 3;
    background: #161b22;
    border-top: solid #30363d;
    padding: 0 1;
}

#user-input {
    width: 1fr;
    background: #21262d;
    border: solid #30363d;
    color: #e6edf3;
    height: 3;
}
#user-input:focus { border: solid #58a6ff; }

#send-btn   { width: 9; background: #238636; color: white; border: none; margin-left: 1; height: 3; }
#send-btn:hover    { background: #2ea043; }
#send-btn:disabled { background: #21262d; color: #6e7681; }

#model-select { width: 100%; background: #21262d; border: solid #30363d; color: #e6edf3; margin-bottom: 1; }
#gguf-input   { width: 100%; background: #21262d; border: solid #30363d; color: #c9d1d9; }

#load-btn  { width: 100%; background: #1f6feb; color: white; border: none; margin-top: 1; }
#load-btn:hover { background: #388bfd; }

#clear-btn { width: 100%; background: #6e2020; color: white; border: none; margin-top: 1; }
#clear-btn:hover { background: #da3633; }

#sys-prompt { height: 5; width: 100%; background: #21262d; border: solid #30363d; color: #c9d1d9; margin-top: 1; }

.slabel { color: #8b949e; text-style: bold; margin-top: 1; margin-bottom: 1; }
#status  { color: #f0883e; margin-top: 1; }
"""


# ─── Metrics Widget ───────────────────────────────────────────────────────────

class MetricsWidget(Widget):
    def compose(self) -> ComposeResult:
        yield Static("[bold #8b949e]── Inference ──[/]")
        yield Static("[#6e7681 italic]no results yet[/]", id="tier-badge")
        yield Static("", id="m-tps")
        yield Static("", id="m-tokens")
        yield Static("", id="m-time")
        yield Static("", id="m-ttft")
        yield Static("", id="m-ram")
        yield Static("", id="m-peak")
        yield Static("", id="m-compress")
        yield Static("", id="m-spec")
        yield Static("", id="m-spark")

    def idle(self):
        self.query_one("#tier-badge", Static).update("[#6e7681 italic]awaiting query…[/]")
        for wid in ("m-tps", "m-tokens", "m-time", "m-ttft", "m-ram", "m-peak",
                    "m-compress", "m-spec", "m-spark"):
            self.query_one(f"#{wid}", Static).update("")

    def show_generating(self):
        self.query_one("#tier-badge", Static).update("[#f0883e bold blink]● generating[/]")

    # Renamed from update() to avoid shadowing Widget.update()
    def update_metrics(self, r: PRISMResult):
        _S.tps_history.append(r.tokens_per_sec)

        tc = {"simple": "#3fb950", "medium": "#e3b341", "complex": "#f85149"}[r.tier.value]
        self.query_one("#tier-badge", Static).update(f"[{tc} bold] {r.tier.value.upper()} [/]")

        tps_c = "#3fb950" if r.tokens_per_sec >= 15 else ("#e3b341" if r.tokens_per_sec >= 5 else "#f85149")
        self.query_one("#m-tps", Static).update(
            f"[#8b949e]TPS      [/][{tps_c} bold]{r.tokens_per_sec}[/]"
        )
        self.query_one("#m-tokens", Static).update(f"[#8b949e]Tokens   [/][#58a6ff]{r.tokens_generated}[/]")
        self.query_one("#m-time",   Static).update(f"[#8b949e]Time     [/][#58a6ff]{r.total_sec}s[/]")

        ttft_c = "#3fb950" if r.ttft_sec < 0.5 else ("#e3b341" if r.ttft_sec < 2.0 else "#f85149")
        self.query_one("#m-ttft", Static).update(
            f"[#8b949e]TTFT     [/][{ttft_c}]{r.ttft_sec}s[/]"
        )

        ram_c = "#3fb950" if r.ram_mb < 500 else ("#e3b341" if r.ram_mb < 1500 else "#f85149")
        self.query_one("#m-ram", Static).update(f"[#8b949e]RAM      [/][{ram_c}]{r.ram_mb:.0f}MB[/]")

        peak_c = "#3fb950" if r.peak_ram_mb < 800 else ("#e3b341" if r.peak_ram_mb < 2000 else "#f85149")
        self.query_one("#m-peak", Static).update(
            f"[#8b949e]PeakRAM  [/][{peak_c}]{r.peak_ram_mb:.0f}MB[/]"
        )

        if r.context_compressed:
            pct = round((1 - r.compressed_context_len / max(r.original_context_len, 1)) * 100)
            self.query_one("#m-compress", Static).update(f"[#8b949e]Compress [/][#e3b341]{pct}% saved[/]")
        else:
            self.query_one("#m-compress", Static).update("[#8b949e]Compress [/][#6e7681]–[/]")

        spec = "[#3fb950]✓ on[/]" if r.speculative_used else "[#6e7681]off[/]"
        self.query_one("#m-spec", Static).update(f"[#8b949e]Spec     [/]{spec}")

        # TPS sparkline
        if _S.tps_history:
            mx = max(_S.tps_history) or 1
            spark = "".join(_SPARK[min(int(v / mx * 7), 7)] for v in _S.tps_history)
            self.query_one("#m-spark", Static).update(f"[#8b949e]TPS hist [/][#58a6ff]{spark}[/]")


# ─── System Monitor Widget ────────────────────────────────────────────────────

class SystemMonitor(Widget):
    """Live CPU%, RAM, memory pressure, chip info. Updates every 2s."""

    _timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Static("[bold #8b949e]── System ──[/]")
        yield Static("", id="sm-chip")
        yield Static("", id="sm-cpu")
        yield Static("", id="sm-ram")
        yield Static("", id="sm-pressure")
        yield Static("", id="sm-cores")
        yield Static("", id="sm-bat")
        yield Static("", id="sm-model")

    def on_mount(self):
        cores_p = psutil.cpu_count(logical=False) or 1
        cores_l = psutil.cpu_count(logical=True) or cores_p
        mem = psutil.virtual_memory()
        self.query_one("#sm-chip", Static).update(
            f"[#8b949e]Chip  [/][#c9d1d9]{_CHIP}[/]"
        )
        self.query_one("#sm-cores", Static).update(
            f"[#8b949e]Cores [/][#58a6ff]{cores_p}P / {cores_l}L[/]"
        )
        self.query_one("#sm-model", Static).update(
            f"[#8b949e]Model [/][#6e7681]{_S.engine_name or 'none'}[/]"
        )
        self._refresh_stats()
        self._timer = self.set_interval(2.0, self._refresh_stats)

    def _refresh_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1024**3
        total_gb = mem.total / 1024**3
        pct = mem.percent

        # CPU bar
        cpu_bar_len = int(cpu / 5)  # 20-char bar
        cpu_bar = "█" * cpu_bar_len + "░" * (20 - cpu_bar_len)
        cpu_c = "#3fb950" if cpu < 50 else ("#e3b341" if cpu < 80 else "#f85149")
        self.query_one("#sm-cpu", Static).update(
            f"[#8b949e]CPU   [/][{cpu_c}]{cpu:5.1f}%[/]"
        )

        # RAM bar
        ram_bar_len = int(pct / 5)
        ram_bar = "█" * ram_bar_len + "░" * (20 - ram_bar_len)
        ram_c = "#3fb950" if pct < 60 else ("#e3b341" if pct < 85 else "#f85149")
        self.query_one("#sm-ram", Static).update(
            f"[#8b949e]RAM   [/][{ram_c}]{used_gb:.1f}[/][#8b949e]/{total_gb:.0f}GB[/]"
        )

        # Memory pressure label
        if pct < 60:
            pressure = ("[#3fb950]● Normal[/]", "#3fb950")
        elif pct < 80:
            pressure = ("[#e3b341]● Warning[/]", "#e3b341")
        else:
            pressure = ("[#f85149]● Critical — swap active[/]", "#f85149")
        self.query_one("#sm-pressure", Static).update(
            f"[#8b949e]Press [/]{pressure[0]}"
        )

        # Battery (if available)
        try:
            bat = psutil.sensors_battery()
            if bat:
                plug = "⚡" if bat.power_plugged else "🔋"
                bat_c = "#3fb950" if bat.percent > 50 else ("#e3b341" if bat.percent > 20 else "#f85149")
                self.query_one("#sm-bat", Static).update(
                    f"[#8b949e]Bat   [/][{bat_c}]{bat.percent:.0f}%[/] {plug}"
                )
        except Exception:
            pass

        # Update model name if changed
        self.query_one("#sm-model", Static).update(
            f"[#8b949e]Model [/][#c9d1d9]{_S.engine_name or 'none'}[/]"
        )


# ─── Main App ─────────────────────────────────────────────────────────────────

class PRISMApp(App):
    TITLE = "PRISM v3  —  Local AI Chat"
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+m", "toggle_panels",  "Metrics"),
        Binding("ctrl+l", "clear_chat",     "Clear"),
        Binding("ctrl+q", "quit",           "Quit"),
        Binding("escape", "refocus",        "Input"),
    ]

    _busy = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("MLX model (HuggingFace)", classes="slabel")
                yield Select([(lbl, val) for lbl, val in MLX_PRESETS], id="model-select",
                             value=MLX_PRESETS[0][1])
                yield Static("or GGUF path (7B–70B):", classes="slabel")
                yield Input(placeholder="./models/llama-70b-q2.gguf", id="gguf-input")
                yield Button("Load Model", id="load-btn", variant="primary")
                yield Static("", id="status")
                yield Static("System prompt", classes="slabel")
                yield TextArea(DEFAULT_SYSTEM, id="sys-prompt")
                yield Button("Clear Chat", id="clear-btn", variant="error")

            # Chat
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
                with Horizontal(id="input-row"):
                    yield Input(placeholder="Type message… (Enter to send)", id="user-input")
                    yield Button("Send", id="send-btn", variant="success")

            # Right panel: metrics + sysmon
            with Vertical(id="right-panel"):
                yield MetricsWidget(id="metrics-panel")
                yield SystemMonitor(id="sysmon-panel")

        yield Footer()

    def on_mount(self):
        self.query_one("#user-input", Input).focus()
        self._syslog("[dim]Select a model and click Load to begin.[/dim]")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _syslog(self, msg: str):
        self.query_one("#chat-log", RichLog).write(Text.from_markup(msg))

    def _log_user(self, text: str):
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup("\n[bold #58a6ff]You[/]"))
        log.write(Text(text))

    def _log_assistant_start(self):
        self.query_one("#chat-log", RichLog).write(Text.from_markup("\n[bold #3fb950]Assistant[/]"))

    def _log_token(self, tok: str):
        self.query_one("#chat-log", RichLog).write(Text(tok), shrink=False)

    def _log_done(self, r: PRISMResult):
        self.query_one("#chat-log", RichLog).write(Text.from_markup(
            f"\n[dim #6e7681]── {r.tier.value} | {r.tokens_per_sec} TPS | "
            f"{r.total_sec}s | {r.tokens_generated} tok ──[/dim]\n"
        ))

    # ── Button events ─────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "load-btn":
            self._load()
        elif event.button.id == "send-btn":
            self._send()
        elif event.button.id == "clear-btn":
            self._clear()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "user-input":
            self._send()

    # ── Load ─────────────────────────────────────────────────────────────────

    def _load(self):
        if _S.loading:
            return
        gguf = self.query_one("#gguf-input", Input).value.strip()
        sel  = self.query_one("#model-select", Select).value
        mode     = "gguf" if gguf else "mlx"
        model_id = "" if gguf else str(sel)

        if mode == "mlx" and not model_id:
            self.query_one("#status", Static).update("[red]Enter a model ID[/]")
            return

        _S.loading = True
        self.query_one("#status", Static).update("[yellow]Loading…[/]")
        self._syslog(f"[yellow]Loading {'GGUF: ' + gguf if gguf else model_id}…[/]")

        def done():
            self.call_from_thread(
                lambda: self.query_one("#status", Static).update(f"[green]✓ {_S.engine_name}[/]")
            )
            self.call_from_thread(
                lambda: self._syslog(f"[green]Loaded: {_S.engine_name}[/]")
            )
            self.call_from_thread(lambda: self.query_one("#user-input", Input).focus())

        def err(e: str):
            self.call_from_thread(
                lambda: self.query_one("#status", Static).update(f"[red]{e[:40]}[/]")
            )
            self.call_from_thread(lambda: self._syslog(f"[red]Load error: {e}[/]"))

        threading.Thread(
            target=_load_thread, args=(mode, model_id, gguf, "", done, err), daemon=True
        ).start()

    # ── Send ─────────────────────────────────────────────────────────────────

    def _send(self):
        if _S.engine is None:
            self._syslog("[red]Load a model first![/]")
            return
        if self._busy:
            return
        inp    = self.query_one("#user-input", Input)
        prompt = inp.value.strip()
        if not prompt:
            return

        inp.clear()
        sys_p = self.query_one("#sys-prompt", TextArea).text.strip()

        self._set_busy(True)
        self.query_one("#metrics-panel", MetricsWidget).show_generating()
        _S.append_history("user", prompt)
        self._log_user(prompt)
        self._log_assistant_start()

        threading.Thread(target=self._gen_thread, args=(prompt, sys_p), daemon=True).start()

    def _gen_thread(self, prompt: str, sys_p: str):
        try:
            parts: list[str] = []
            final: Optional[PRISMResult] = None
            # prior_history excludes current user turn (last item just appended in _send)
            prior_history = _S.history[:-1]
            # Batch tokens → send to UI every 8 tokens to reduce thread-switch overhead
            _batch: list[str] = []

            def _flush_batch():
                if _batch:
                    self.call_from_thread(self._log_token, "".join(_batch))
                    _batch.clear()

            with _engine_lock:
                engine = _S.engine
            for tok, meta in engine.generate_stream(prompt, sys_p, history=prior_history):
                if meta is None:
                    parts.append(tok)
                    _batch.append(tok)
                    if len(_batch) >= 8:
                        _flush_batch()
                else:
                    _flush_batch()
                    final = meta

            if final:
                _S.last_result = final
                _S.append_history("assistant", "".join(parts))
                self.call_from_thread(self._log_done, final)
                # Use update_metrics (not update) to avoid Widget.update() conflict
                self.call_from_thread(
                    self.query_one("#metrics-panel", MetricsWidget).update_metrics, final
                )
        except Exception as e:
            self.call_from_thread(self._syslog, f"[red]Error: {e}[/]")
        finally:
            # Reactive mutation must happen on the main thread
            self.call_from_thread(self._set_busy, False)

    def _set_busy(self, val: bool):
        """Must be called from main thread (reactive attribute)."""
        self._busy = val
        self.query_one("#send-btn", Button).disabled = val

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear(self):
        _S.history.clear()
        _S.tps_history.clear()
        if _S.engine and hasattr(_S.engine, "kv_cache"):
            _S.engine.kv_cache.reset()
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#metrics-panel", MetricsWidget).idle()
        self._syslog("[dim]Chat cleared.[/dim]")

    def action_clear_chat(self): self._clear()
    def action_refocus(self): self.query_one("#user-input", Input).focus()

    def action_toggle_sidebar(self):
        sb = self.query_one("#sidebar")
        sb.toggle_class("hidden")

    def action_toggle_panels(self):
        rp = self.query_one("#right-panel")
        rp.toggle_class("hidden")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PRISM v3 Terminal UI")
    ap.add_argument("--model", default="", help="MLX HuggingFace model ID")
    ap.add_argument("--gguf",  default="", help="Path to GGUF file (70B support)")
    ap.add_argument("--draft", default="", help="Draft model for speculative decoding")
    args = ap.parse_args()

    app = PRISMApp()

    if args.model or args.gguf:
        def auto_load():
            time.sleep(0.8)
            _S.loading = True
            mode = "gguf" if args.gguf else "mlx"

            def done():
                app.call_from_thread(
                    lambda: app.query_one("#status", Static).update(f"[green]✓ {_S.engine_name}[/]")
                )
                app.call_from_thread(
                    lambda: app._syslog(f"[green]Loaded: {_S.engine_name}[/]")
                )

            def err(e):
                app.call_from_thread(lambda: app._syslog(f"[red]Auto-load failed: {e}[/]"))

            _load_thread(mode, args.model, args.gguf, args.draft, done, err)

        threading.Thread(target=auto_load, daemon=True).start()

    app.run()


if __name__ == "__main__":
    main()
