"""Full-screen snapshots, not an emulated remote PTY or reconstructed byte stream."""
from __future__ import annotations

import asyncio
import hashlib
import time
import unicodedata
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


KEYS = {
    "enter": "Enter", "c-m": "Enter", "c-j": "Enter",
    "tab": "Tab", "c-i": "Tab", "s-tab": "Shift+Tab",
    "backspace": "Backspace", "c-h": "Backspace",
    "escape": "Escape",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
    "delete": "Delete", "insert": "Insert", "c-@": "Control+Space",
    "c-\\": "Control+\\", "c-^": "Control+^", "c-_": "Control+_",
}
for _letter in "abcdefghijklmnopqrstuvwxyz":
    KEYS.setdefault(f"c-{_letter}", f"Control+{_letter}")
for _number in range(1, 25):
    KEYS[f"f{_number}"] = f"F{_number}"
for _modifier, _browser_modifier in (("s", "Shift"), ("c", "Control"), ("c-s", "Control+Shift")):
    for _name in ("up", "down", "left", "right", "home", "end", "delete", "insert", "pageup", "pagedown"):
        KEYS[f"{_modifier}-{_name}"] = f"{_browser_modifier}+{KEYS[_name]}"


def safe_screen(text: str) -> str:
    """Never let OCR emit escape sequences or invisible directional controls."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        char if char in "\n\t" or not unicodedata.category(char).startswith("C") else "\ufffd"
        for char in text
    ).expandtabs(8)


@dataclass
class Screen:
    text: str = "Waiting for first OCR frame..."
    captured: float | None = None
    latency: float = 0.0
    busy: bool = False


class TerminalMirror:
    def __init__(self, browser, ocr, *, interval: float = 0.5, input=None, output=None):
        self.browser = browser
        self.ocr = ocr
        self.interval = interval
        self.screen = Screen()
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=4096)
        self.tasks: list[asyncio.Task] = []
        self.bindings = KeyBindings()
        self._bind_keys()
        body = FormattedTextControl(lambda: self.screen.text, focusable=True, show_cursor=False)
        status = FormattedTextControl(self._status)
        self.app = Application(
            layout=Layout(HSplit([
                Window(body, wrap_lines=False, always_hide_cursor=True),
                Window(status, height=1, style="class:status"),
            ])),
            key_bindings=self.bindings,
            full_screen=True,
            mouse_support=False,
            style=Style.from_dict({"status": "reverse"}),
            refresh_interval=0.25,
            input=input,
            output=output,
        )
        self.app.timeoutlen = 0.15
        self.app.ttimeoutlen = 0.05

    def _status(self):
        age = "no frame" if self.screen.captured is None else f"frame {time.monotonic() - self.screen.captured:.1f}s old"
        state = " | OCR running" if self.screen.busy else ""
        if not self.screen.text.strip():
            state += " | no text recognized"
        return f" OCR mirror (approximate) | {age} | OCR {self.screen.latency:.1f}s{state} | queued {self.queue.qsize()} | Ctrl+] exit "

    def _enqueue(self, kind: str, value: str):
        try:
            self.queue.put_nowait((kind, value))
        except asyncio.QueueFull:
            self.app.exit(exception=RuntimeError("Keyboard queue overflow: stopped. Some input may already have reached the VM; inspect it before continuing."))

    def _bind_keys(self):
        @self.bindings.add("c-]", eager=True)
        def exit_local(event):
            event.app.exit()

        @self.bindings.add(Keys.BracketedPaste)
        def paste(event):
            self._enqueue("paste", event.data)

        @self.bindings.add(Keys.Any)
        def character(event):
            if event.data and all(char.isprintable() for char in event.data):
                self._enqueue("text", event.data)

        for name, browser_key in KEYS.items():
            self.bindings.add(name)(self._key_handler(browser_key))
            if browser_key != "Escape":
                self.bindings.add("escape", name)(self._key_handler("Alt+" + browser_key))

        @self.bindings.add("escape", Keys.Any)
        def alt_character(event):
            if len(event.data) == 1 and event.data.isascii() and event.data.isprintable():
                self._enqueue("key", "Alt+" + event.data)

    def _key_handler(self, key):
        def send(event):
            self._enqueue("key", key)
        return send

    async def _forward(self):
        while True:
            kind, value = await self.queue.get()
            try:
                if kind == "text":
                    await self.browser.send_text(value)
                elif kind == "paste":
                    await self.browser.send_paste(value)
                else:
                    await self.browser.send_key(value)
            finally:
                self.queue.task_done()

    async def _capture(self):
        previous = None
        while True:
            started = time.monotonic()
            png = await self.browser.screenshot()
            digest = hashlib.sha256(png).digest()
            if digest != previous:
                self.screen.busy = True
                self.app.invalidate()
                text = await self.ocr.recognize(png)
                self.screen.text = safe_screen(text)
                self.screen.captured = started
                self.screen.latency = time.monotonic() - started
                self.screen.busy = False
                previous = digest
                self.app.invalidate()
            await asyncio.sleep(max(0.01, self.interval - (time.monotonic() - started)))

    def _task_finished(self, task):
        if not task.cancelled():
            error = task.exception()
            if error is not None and self.app.is_running:
                self.app.exit(exception=error)

    async def run(self):
        def start():
            self.tasks = [asyncio.create_task(self._capture()), asyncio.create_task(self._forward())]
            for task in self.tasks:
                task.add_done_callback(self._task_finished)
        try:
            await self.app.run_async(pre_run=start, handle_sigint=False)
        finally:
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
