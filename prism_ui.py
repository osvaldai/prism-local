#!/usr/bin/env python3
"""
PRISM UI — Terminal chat interface with live metrics.

Features:
  - Model picker: MLX (HuggingFace) or GGUF (local file, supports 70B)
  - Chat with real-time token streaming
  - Live metrics: TPS, RAM, tier, compression ratio, speculative flag
  - System prompt editor
  - Conversation history

Run:
    python prism_ui.py
    python prism_ui.py --model mlx-community/gemma-3-4b-it-4bit
    python prism_ui.py --gguf ./models/llama-70b-q2.gguf

Install deps first:
    pip install textual
"""
import argparse
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Footer, Header, Input, RichLog, Select, Static, TextArea,
)
from rich.text import Text

from prism_engine_v2 import PRISMResult, Tier, profile_hardware


# ─── Model presets ───────────────────────────────────────────────────────────

MLX_PRESETS = [
    ("Gemma 3 4B INT4  — 3.3 GB (recommended)",  "mlx-community/gemma-3-4b-it-4bit"),
    ("Gemma 3 12B INT4 — 6.5 GB (16GB RAM)",      "mlx-community/gemma-3-12b-it-4bit"),
    ("Llama 3.1 8B INT4 — 4.7 GB",               "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
    ("Mistral 7B INT4  — 4.1 GB",                "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
    ("Qwen 2.5 7B INT4 — 4.3 GB",               "mlx-community/Qwen2.5-7B-Instruct-4bit"),
]

DEFAULT_SYSTEM = "You are a helpful, concise assistant."


# ─── App state ───────────────────────────────────────────────────────────────

@dataclass
class _State:
    engine: object = None
    engine_name: str = ""
    loading: bool = False
    last_result: Optional[PRISMResult] = None
    history: list[dict] = field(default_factory=list)


_S = _State()


# ─── Background loader ────────────────────────────────────────────────────────

def _load_thread(mode: str, model_id: str, gguf_path: str, draft_id: str, on_done, on_err):
    try:
        if mode == "mlx":
            from prism_engine_v2 import load_prism
            _S.engine = load_prism(model_id, draft_model_id=draft_id or None)
            _S.engine_name = model_id.split("/")[-1]
        else:
            from llama_engine import LlamaEngine
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

#sidebar {
    width: 30;
    background: #161b22;
    border-right: solid #30363d;
    padding: 1 1;
}

#chat-panel { width: 1fr; background: #0d1117; }

#metrics-panel {
    width: 25;
    background: #161b22;
    border-left: solid #30363d;
    padding: 1 1;
}

#chat-log {
    height: 1fr;
    border: solid #21262d;
    background: #010409;
    padding: 0 1;
}

#input-row {
    height: 5;
    background: #161b22;
    border-top: solid #30363d;
    padding: 1 1;
}

#user-input {
    width: 1fr;
    background: #21262d;
    border: solid #30363d;
    color: #e6edf3;
}
#user-input:focus { border: solid #58a6ff; }

#send-btn   { width: 9; background: #238636; color: white; border: none; margin-left: 1; }
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

#tier-badge  { text-align: center; text-style: bold; margin-bottom: 1; height: 1; }
.mval        { color: #58a6ff; text-style: bold; }
.mval-g      { color: #3fb950; text-style: bold; }
.mval-y      { color: #e3b341; text-style: bold; }
.mval-r      { color: #f85149; text-style: bold; }
"""


# ─── Metrics Widget ───────────────────────────────────────────────────────────

class MetricsWidget(Widget):
    DEFAULT_CSS = ""

    def compose(self) -> ComposeResult:
        yield Static("[bold #8b949e]── Metrics ──[/]")
        yield Static("", id="tier-badge")
        yield Static("", id="m-tps")
        yield Static("", id="m-tokens")
        yield Static("", id="m-time")
        yield Static("", id="m-ram")
        yield Static("", id="m-compress")
        yield Static("", id="m-spec")
        yield Static("", id="m-hw")

    def on_mount(self):
        hw = profile_hardware()
        self.query_one("#m-hw", Static).update(
            f"[#8b949e]RAM  [/][#58a6ff]{hw.total_ram_gb:.0f}GB[/] "
            f"[#8b949e]free[/] [#3fb950]{hw.free_ram_gb:.1f}GB[/]"
        )
        self.idle()

    def idle(self):
        self.query_one("#tier-badge", Static).update("[#6e7681 italic]awaiting query…[/]")
        for wid in ("m-tps", "m-tokens", "m-time", "m-ram", "m-compress", "m-spec"):
            self.query_one(f"#{wid}", Static).update("")

    def generating(self):
        self.query_one("#tier-badge", Static).update("[#f0883e bold blink]● generating[/]")

    def update(self, r: PRISMResult):  # noqa: A003
        tc = {"simple": "#3fb950", "medium": "#e3b341", "complex": "#f85149"}[r.tier.value]
        self.query_one("#tier-badge", Static).update(f"[{tc} bold] {r.tier.value.upper()} [/]")

        tps_c = "#3fb950" if r.tokens_per_sec >= 15 else ("#e3b341" if r.tokens_per_sec >= 5 else "#f85149")
        self.query_one("#m-tps", Static).update(
            f"[#8b949e]TPS     [/][{tps_c} bold]{r.tokens_per_sec}[/]"
        )
        self.query_one("#m-tokens", Static).update(f"[#8b949e]Tokens  [/][#58a6ff]{r.tokens_generated}[/]")
        self.query_one("#m-time",   Static).update(f"[#8b949e]Time    [/][#58a6ff]{r.total_sec}s[/]")
        ram_c = "#3fb950" if r.ram_mb < 500 else ("#e3b341" if r.ram_mb < 1500 else "#f85149")
        self.query_one("#m-ram",    Static).update(f"[#8b949e]RAM     [/][{ram_c}]{r.ram_mb:.0f}MB[/]")

        if r.context_compressed:
            pct = round((1 - r.compressed_context_len / max(r.original_context_len, 1)) * 100)
            self.query_one("#m-compress", Static).update(f"[#8b949e]Compress[/] [#e3b341]{pct}% saved[/]")
        else:
            self.query_one("#m-compress", Static).update("[#8b949e]Compress[/] [#6e7681]–[/]")

        spec = "[#3fb950]✓ on[/]" if r.speculative_used else "[#6e7681]off[/]"
        self.query_one("#m-spec", Static).update(f"[#8b949e]Spec    [/]{spec}")

        hw = profile_hardware()
        self.query_one("#m-hw", Static).update(
            f"[#8b949e]RAM  [/][#58a6ff]{hw.total_ram_gb:.0f}GB[/] "
            f"[#8b949e]free[/] [#3fb950]{hw.free_ram_gb:.1f}GB[/]"
        )


# ─── App ─────────────────────────────────────────────────────────────────────

class PRISMApp(App):
    TITLE = "PRISM v2  —  Local AI Chat"
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "refocus", "Focus input"),
    ]

    _busy = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            # ── Sidebar ──────────────────────────────────────────────────────
            with Vertical(id="sidebar"):
                yield Static("MLX model", classes="slabel")
                yield Select(
                    [(lbl, val) for lbl, val in MLX_PRESETS],
                    id="model-select",
                    value=MLX_PRESETS[0][1],
                )
                yield Static("or GGUF path (70B)", classes="slabel")
                yield Input(placeholder="./models/llama-70b-q2.gguf", id="gguf-input")
                yield Button("Load Model", id="load-btn", variant="primary")
                yield Static("", id="status")
                yield Static("System prompt", classes="slabel")
                yield TextArea(DEFAULT_SYSTEM, id="sys-prompt")
                yield Button("Clear Chat", id="clear-btn", variant="error")

            # ── Chat ─────────────────────────────────────────────────────────
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
                with Horizontal(id="input-row"):
                    yield Input(placeholder="Type message… (Enter to send)", id="user-input")
                    yield Button("Send", id="send-btn", variant="success")

            # ── Metrics ──────────────────────────────────────────────────────
            yield MetricsWidget(id="metrics-panel")

        yield Footer()

    def on_mount(self):
        self.query_one("#user-input", Input).focus()
        self._syslog("[dim]Select a model and click Load to begin.[/]")

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
            f"{r.total_sec}s | {r.tokens_generated} tok ──[/]\n"
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
        if _S.loading or _S.loading:
            return
        gguf = self.query_one("#gguf-input", Input).value.strip()
        sel  = self.query_one("#model-select", Select).value

        mode     = "gguf" if gguf else "mlx"
        model_id = "" if gguf else str(sel)

        if mode == "mlx" and not model_id:
            self.query_one("#status", Static).update("[red]Enter a model ID[/]")
            return

        _S.loading = True
        self.query_one("#status", Static).update("[yellow blink]Loading…[/]")
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
                lambda: self.query_one("#status", Static).update(f"[red]Error: {e[:35]}[/]")
            )
            self.call_from_thread(lambda: self._syslog(f"[red]Load error: {e}[/]"))

        threading.Thread(
            target=_load_thread,
            args=(mode, model_id, gguf, "", done, err),
            daemon=True,
        ).start()

    # ── Send ─────────────────────────────────────────────────────────────────

    def _send(self):
        if _S.engine is None:
            self._syslog("[red]Load a model first![/]")
            return
        if self._busy:
            return
        inp = self.query_one("#user-input", Input)
        prompt = inp.value.strip()
        if not prompt:
            return

        inp.clear()
        sys_p = self.query_one("#sys-prompt", TextArea).text.strip()

        self._busy = True
        self.query_one("#send-btn", Button).disabled = True
        self.query_one("#metrics-panel", MetricsWidget).generating()

        _S.history.append({"role": "user", "content": prompt})
        self._log_user(prompt)
        self._log_assistant_start()

        threading.Thread(target=self._gen_thread, args=(prompt, sys_p), daemon=True).start()

    def _gen_thread(self, prompt: str, sys_p: str):
        try:
            parts: list[str] = []
            final: Optional[PRISMResult] = None

            for tok, meta in _S.engine.generate_stream(prompt, sys_p):
                if meta is None:
                    parts.append(tok)
                    self.call_from_thread(self._log_token, tok)
                else:
                    final = meta

            if final:
                _S.last_result = final
                _S.history.append({"role": "assistant", "content": "".join(parts)})
                self.call_from_thread(self._log_done, final)
                self.call_from_thread(self.query_one("#metrics-panel", MetricsWidget).update, final)
        except Exception as e:
            self.call_from_thread(self._syslog, f"[red]Error: {e}[/]")
        finally:
            self._busy = False
            self.call_from_thread(
                lambda: setattr(self.query_one("#send-btn", Button), "disabled", False)
            )

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear(self):
        _S.history.clear()
        if _S.engine and hasattr(_S.engine, "kv_cache"):
            _S.engine.kv_cache.reset()
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#metrics-panel", MetricsWidget).idle()
        self._syslog("[dim]Chat cleared.[/]")

    def action_clear_chat(self): self._clear()
    def action_refocus(self): self.query_one("#user-input", Input).focus()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PRISM v2 Terminal UI")
    ap.add_argument("--model", default="", help="MLX HuggingFace model ID")
    ap.add_argument("--gguf",  default="", help="Path to GGUF model file (70B support)")
    ap.add_argument("--draft", default="", help="Draft model for speculative decoding")
    args = ap.parse_args()

    app = PRISMApp()

    if args.model or args.gguf:
        def auto_load():
            import time; time.sleep(0.8)
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
