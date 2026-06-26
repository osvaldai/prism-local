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
Screen { background: #080b10; }
Header { background: #0c1120; color: #8a9abb; height: 3; }
Footer { background: #0a0e18; color: #3a4a62; }

#layout { height: 1fr; }

/* ── Sidebar ── */
#sidebar {
    width: 22;
    min-width: 22;
    background: #0a0e18;
    border-right: solid #18243a;
    padding: 1 1;
}
#sidebar.hidden { display: none; }

/* ── Chat ── */
#chat-panel { width: 1fr; background: #080b10; }

/* ── Right panel ── */
#right-panel {
    width: 25;
    min-width: 25;
    background: #0a0e18;
    border-left: solid #18243a;
}
#right-panel.hidden { display: none; }

#metrics-panel {
    height: auto;
    padding: 1 1;
    border-bottom: solid #18243a;
}

#sysmon-panel {
    height: 1fr;
    padding: 1 1;
}

/* ── Chat log ── */
#chat-log {
    height: 1fr;
    background: #080b10;
    padding: 1 3;
    border: none;
}

/* ── Input row ── */
#input-row {
    height: 4;
    background: #0c1120;
    border-top: solid #18243a;
    padding: 0 1;
}

#user-input {
    width: 1fr;
    background: #111826;
    border: solid #1e2d45;
    color: #c8d4ea;
    height: 4;
}
#user-input:focus { border: solid #3d7eff; }

#send-btn   { width: 9; background: #142a50; color: #3d7eff; border: solid #1e3a6a; margin-left: 1; height: 4; }
#send-btn:hover    { background: #1a3860; }
#send-btn:disabled { background: #111826; color: #253248; }

#model-select { width: 100%; background: #111826; border: solid #1e2d45; color: #8a9abb; margin-bottom: 1; }
#gguf-input   { width: 100%; background: #111826; border: solid #1e2d45; color: #8a9abb; }

#load-btn  { width: 100%; background: #142a50; color: #3d7eff; border: none; margin-top: 1; }
#load-btn:hover { background: #1a3860; }

#clear-btn { width: 100%; background: #2a1010; color: #ff5555; border: none; margin-top: 1; }
#clear-btn:hover { background: #401818; }

#sys-prompt { height: 5; width: 100%; background: #111826; border: solid #1e2d45; color: #8a9abb; margin-top: 1; }

.slabel { color: #2e4060; text-style: bold; margin-top: 1; margin-bottom: 0; }
#status  { color: #e8963a; margin-top: 1; }
"""


# ─── Metrics Widget ───────────────────────────────────────────────────────────

class MetricsWidget(Widget):
    def compose(self) -> ComposeResult:
        yield Static("[bold #2e4060]◈ INFERENCE[/]")
        yield Static("[#253248 italic]awaiting query…[/]", id="tier-badge")
        yield Static("", id="m-tps")
        yield Static("", id="m-spark")
        yield Static("", id="m-ttft")
        yield Static("", id="m-tokens")
        yield Static("", id="m-time")
        yield Static("[bold #2e4060]◈ MEMORY[/]", id="m-mem-hdr")
        yield Static("", id="m-ram")
        yield Static("", id="m-peak")
        yield Static("[bold #2e4060]◈ CONTEXT[/]", id="m-ctx-hdr")
        yield Static("", id="m-compress")
        yield Static("", id="m-spec")

    def idle(self):
        self.query_one("#tier-badge", Static).update("[#253248 italic]awaiting query…[/]")
        for wid in ("m-tps", "m-spark", "m-ttft", "m-tokens", "m-time",
                    "m-ram", "m-peak", "m-compress", "m-spec"):
            self.query_one(f"#{wid}", Static).update("")

    def show_generating(self):
        self.query_one("#tier-badge", Static).update("[#e8963a bold]⬡ generating…[/]")

    def update_metrics(self, r: PRISMResult):
        _S.tps_history.append(r.tokens_per_sec)

        tc = {"simple": "#2dba52", "medium": "#d4a030", "complex": "#e84040"}[r.tier.value]
        tier_icon = {"simple": "○", "medium": "◑", "complex": "●"}[r.tier.value]
        self.query_one("#tier-badge", Static).update(
            f"[{tc} bold]{tier_icon} {r.tier.value.upper()}[/]"
        )

        tps_c = "#2dba52" if r.tokens_per_sec >= 15 else ("#d4a030" if r.tokens_per_sec >= 5 else "#e84040")
        self.query_one("#m-tps", Static).update(
            f"[#3a4a62]tps  [/][{tps_c} bold]{r.tokens_per_sec:5.1f}[/]"
        )

        if _S.tps_history:
            mx = max(_S.tps_history) or 1
            spark = "".join(_SPARK[min(int(v / mx * 7), 7)] for v in _S.tps_history)
            self.query_one("#m-spark", Static).update(f"[#3a4a62]hist [/][#3d7eff]{spark}[/]")

        ttft_c = "#2dba52" if r.ttft_sec < 0.5 else ("#d4a030" if r.ttft_sec < 2.0 else "#e84040")
        self.query_one("#m-ttft", Static).update(
            f"[#3a4a62]ttft [/][{ttft_c}]{r.ttft_sec:.2f}s[/]"
        )
        self.query_one("#m-tokens", Static).update(f"[#3a4a62]tok  [/][#8a9abb]{r.tokens_generated}[/]")
        self.query_one("#m-time",   Static).update(f"[#3a4a62]time [/][#8a9abb]{r.total_sec}s[/]")

        ram_c = "#2dba52" if r.ram_mb < 500 else ("#d4a030" if r.ram_mb < 1500 else "#e84040")
        self.query_one("#m-ram", Static).update(f"[#3a4a62]rss  [/][{ram_c}]{r.ram_mb:.0f}[#3a4a62]MB[/]")

        peak_c = "#2dba52" if r.peak_ram_mb < 800 else ("#d4a030" if r.peak_ram_mb < 2000 else "#e84040")
        self.query_one("#m-peak", Static).update(
            f"[#3a4a62]peak [/][{peak_c}]{r.peak_ram_mb:.0f}[#3a4a62]MB[/]"
        )

        if r.context_compressed:
            pct = round((1 - r.compressed_context_len / max(r.original_context_len, 1)) * 100)
            self.query_one("#m-compress", Static).update(
                f"[#3a4a62]cmp  [/][#d4a030]−{pct}%[/]"
            )
        else:
            self.query_one("#m-compress", Static).update("[#3a4a62]cmp  [/][#253248]none[/]")

        spec = "[#2dba52]✓[/]" if r.speculative_used else "[#253248]✗[/]"
        self.query_one("#m-spec", Static).update(f"[#3a4a62]spec [/]{spec}")


# ─── System Monitor Widget ────────────────────────────────────────────────────

def _bar(pct: float, width: int = 12, c_ok="#2dba52", c_warn="#d4a030", c_bad="#e84040") -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    color = c_ok if pct < 60 else (c_warn if pct < 85 else c_bad)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/]"


class SystemMonitor(Widget):
    """Live CPU%, RAM, memory pressure, chip info. Updates every 2s."""

    _timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Static("[bold #2e4060]◈ SYSTEM[/]")
        yield Static("", id="sm-chip")
        yield Static("", id="sm-cores")
        yield Static("", id="sm-cpu")
        yield Static("", id="sm-ram")
        yield Static("", id="sm-pressure")
        yield Static("", id="sm-bat")
        yield Static("", id="sm-model")

    def on_mount(self):
        cores_p = psutil.cpu_count(logical=False) or 1
        cores_l = psutil.cpu_count(logical=True) or cores_p
        chip_short = _CHIP.replace("Apple ", "").split(" ")[0:3]
        self.query_one("#sm-chip", Static).update(
            f"[#3a4a62]chip [/][#8a9abb]{' '.join(chip_short)}[/]"
        )
        self.query_one("#sm-cores", Static).update(
            f"[#3a4a62]core [/][#3d7eff]{cores_p}P[/][#3a4a62]+[/][#3d7eff]{cores_l - cores_p}E[/]"
        )
        self._refresh_stats()
        self._timer = self.set_interval(2.0, self._refresh_stats)

    def _refresh_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1024**3
        total_gb = mem.total / 1024**3
        pct = mem.percent

        cpu_c = "#2dba52" if cpu < 50 else ("#d4a030" if cpu < 80 else "#e84040")
        self.query_one("#sm-cpu", Static).update(
            f"[#3a4a62]cpu  [/]{_bar(cpu)} [{cpu_c}]{cpu:4.0f}%[/]"
        )

        ram_c = "#2dba52" if pct < 60 else ("#d4a030" if pct < 85 else "#e84040")
        self.query_one("#sm-ram", Static).update(
            f"[#3a4a62]ram  [/]{_bar(pct)} [{ram_c}]{used_gb:.1f}[#3a4a62]G[/]"
        )

        if pct < 60:
            p_txt = "[#2dba52]● ok[/]"
        elif pct < 85:
            p_txt = "[#d4a030]● warn[/]"
        else:
            p_txt = "[#e84040]● swap![/]"
        self.query_one("#sm-pressure", Static).update(f"[#3a4a62]     [/]{p_txt}")

        try:
            bat = psutil.sensors_battery()
            if bat:
                plug = "⚡" if bat.power_plugged else "○"
                bat_c = "#2dba52" if bat.percent > 50 else ("#d4a030" if bat.percent > 20 else "#e84040")
                self.query_one("#sm-bat", Static).update(
                    f"[#3a4a62]bat  [/]{_bar(bat.percent)} [{bat_c}]{bat.percent:.0f}%[/]{plug}"
                )
        except Exception:
            pass

        self.query_one("#sm-model", Static).update(
            f"[#3a4a62]mdl  [/][#8a9abb]{(_S.engine_name or '—')[:16]}[/]"
        )


# ─── Main App ─────────────────────────────────────────────────────────────────

class PRISMApp(App):
    TITLE = "⬡ PRISM  ·  Local AI"
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
                yield Static("[bold #3a4a62]MLX MODEL[/]", classes="slabel")
                yield Select([(lbl, val) for lbl, val in MLX_PRESETS], id="model-select",
                             value=MLX_PRESETS[0][1])
                yield Static("[bold #3a4a62]GGUF PATH[/]", classes="slabel")
                yield Input(placeholder="./models/llama-70b-q2.gguf", id="gguf-input")
                yield Button("⬡ Load Model", id="load-btn", variant="primary")
                yield Static("", id="status")
                yield Static("[bold #3a4a62]SYSTEM PROMPT[/]", classes="slabel")
                yield TextArea(DEFAULT_SYSTEM, id="sys-prompt")
                yield Button("✕ Clear Chat", id="clear-btn", variant="error")

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
        self._syslog("[#18243a]─────────────────────────────────────────────────────────[/]")
        self._syslog("[#2e4060]  ⬡ PRISM  ·  Progressive Resolution Inference System    [/]")
        self._syslog("[#253248]  Select a model in the sidebar → Load → start chatting. [/]")
        self._syslog("[#253248]  Ctrl+B sidebar  ·  Ctrl+M metrics  ·  Ctrl+L clear     [/]")
        self._syslog("[#18243a]─────────────────────────────────────────────────────────[/]")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _syslog(self, msg: str):
        self.query_one("#chat-log", RichLog).write(Text.from_markup(f"[#253248]{msg}[/]"))

    def _log_user(self, text: str):
        log = self.query_one("#chat-log", RichLog)
        log.write(Text.from_markup(
            "\n[bold #3d7eff]▶ You[/bold #3d7eff]  "
            "[#1e2d45]─────────────────────────────────────[/]"
        ))
        log.write(Text.from_markup(f"[#c8d4ea]  {text}[/]"))

    def _log_assistant_start(self):
        self.query_one("#chat-log", RichLog).write(Text.from_markup(
            "\n[bold #2dba52]◆ PRISM[/bold #2dba52]  "
            "[#18243a]─────────────────────────────────────[/]"
        ))
        self.query_one("#chat-log", RichLog).write(Text.from_markup("  "))

    def _log_token(self, tok: str):
        self.query_one("#chat-log", RichLog).write(Text(tok), shrink=False)

    def _log_done(self, r: PRISMResult):
        tier_c = {"simple": "#2dba52", "medium": "#d4a030", "complex": "#e84040"}[r.tier.value]
        self.query_one("#chat-log", RichLog).write(Text.from_markup(
            f"\n  [#18243a]╌╌ [{tier_c}]{r.tier.value}[/] "
            f"· [#3d7eff]{r.tokens_per_sec}tps[/] "
            f"· [#8a9abb]{r.total_sec}s[/] "
            f"· [#3a4a62]{r.tokens_generated}tok[/] ╌╌[/]\n"
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
