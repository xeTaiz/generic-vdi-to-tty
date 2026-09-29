# VDI terminal bridge

A local terminal interface for an Ubuntu VM that is reachable only through a graphical browser session. Keyboard events go through the browser; screenshots return through OCR. Nothing needs to be installed inside the VM, and the VM needs no network connection.

This is a **screen mirror, not SSH**. It displays the latest recognized screen, not an exact stdout stream. OCR can misread characters, normalize whitespace, or miss output that scrolls away between captures. Do not copy OCR-derived commands or source code into production without checking them.

## Install

Python 3.10+ and Chromium are required. The browser runs on your local workstation. The default Unlimited-OCR backend connects to a separately running vLLM server; it does not download or start the model automatically.

C1
```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/playwright install chromium
```

On Linux, Playwright may also require distribution-specific browser libraries; its installer can report missing dependencies. A visible browser requires a graphical desktop session.

For an explicitly selected, self-contained CPU OCR alternative:

C2
```sh
.venv/bin/python -m pip install -e '.[cpu]'
```

RapidOCR uses models bundled in its Python package. There is no automatic fallback between OCR engines.

## Start with a dedicated browser

Set `VDI_URL` to your website URL. `canvas` is only a starting selector: replace it if the client uses several canvases, a video element, or a different remote-display container.

C3
```sh
.venv/bin/vdi-terminal --url "$VDI_URL" --selector canvas
```

Log in in the opened browser, open the VM terminal, enlarge its monospace font, and keep the terminal visible. Press Enter in the local launcher when ready. It clicks the selected display/crop center to focus input, then enters the local full-screen mirror. With a desktop-wide selector, make sure this point lies inside the terminal.

The persistent browser profile defaults to `~/.local/share/vdi-terminal/browser`; it contains authentication data. Use a dedicated profile, not your everyday Chrome profile. Exiting closes a browser launched this way, which may disconnect the graphical VDI session. The VM's session policy determines whether programs keep running; use an existing terminal multiplexer if available.

To run without a GPU/OCR server, add `--ocr rapidocr` after installing the CPU extra.

## Attach to an existing browser

Chromium must have remote debugging enabled. Keep that endpoint on loopback: it grants control over authenticated browser sessions. Modern Chromium requires a separate user-data directory when enabling remote debugging.

C4
```sh
chromium --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.local/share/vdi-terminal/chrome-cdp"
```

Log in and open the VDI tab in that browser. Set `VDI_TARGET` to a distinctive substring of its URL or title:

C5
```sh
.venv/bin/vdi-terminal --cdp http://127.0.0.1:9222 \
  --target "$VDI_TARGET" --selector canvas
```

The filter must match exactly one tab. The tool does not navigate or close CDP-attached tabs or browsers, and does not automatically choose an arbitrary matching session.

## Terminal region and input

**D1 — Crop and fixed geometry.** `--crop x,y,width,height` selects a terminal rectangle relative to the selected display element, in CSS pixels. `--frame 'iframe-selector'` selects an iframe containing that element. Screenshots use CSS-pixel scale. Keep VDI geometry fixed; local terminal resizing does not resize the remote terminal. Use a high-contrast theme, no transparency, and a local terminal large enough for the recognized screen. A blinking remote cursor can trigger additional OCR passes.

**D2 — Keyboard control.** Printable ASCII, Enter, Tab, Backspace, arrows, navigation keys, function keys, and common Ctrl/Alt combinations are forwarded. Ctrl+C interrupts the remote command; **Ctrl+] exits locally** and is reserved. There is no speculative local echo. Terminal applications cannot represent every physical keyboard combination, and remote layout/browser shortcuts can affect delivery. Keep the remote terminal focused; do not use the browser for unrelated interactions while attached.

**D3 — Paste and noVNC.** Default input uses actual browser keyboard events with short pacing, not DOM text insertion. Local bracketed paste is typed through those events, so embedded newlines submit commands and tabs behave as Tab presses. Review multiline input before pasting. Non-ASCII input is rejected in this mode before that text batch is sent. If you can identify the running noVNC RFB object, pass `--rfb-expression` with its trusted JavaScript expression. It is evaluated in the selected frame; ordinary text/control keys then use the RFB API, and bracketed paste uses the remote clipboard plus Ctrl+Shift+V. Clipboard delivery uses a short best-effort delay, not a verified acknowledgement, and requires clipboard forwarding plus that terminal paste shortcut. No global RFB variable name is assumed. Other VDI clients use the default keyboard path.

## Unlimited-OCR server

Follow the [official vLLM recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR). This is a document-parsing vision-language model, not a terminal-specific OCR engine. It can reflow text or add Markdown formatting; benchmark it against actual terminal screenshots before relying on its output. Its per-frame latency determines how frequently the screen can be read.

The official recipe specifies a dedicated image and custom logits processor. On a machine with Docker, NVIDIA container support, and sufficient free GPU memory:

C6
```sh
docker run --rm --gpus all --ipc host -p 127.0.0.1:8000:8000 \
  vllm/vllm-openai:unlimited-ocr \
  baidu/Unlimited-OCR --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching --mm-processor-cache-gb 0
```

The upstream recipe states at least 8 GB GPU memory. Review the model's remote code before enabling `--trust-remote-code`. The bridge uses the required literal `<image>document parsing.` prompt, `skip_special_tokens=false`, and n-gram processor parameters. It removes recognized layout annotations without intentionally reformatting the remaining text, and rejects truncated generations.

The default endpoint is `http://127.0.0.1:8000/v1`. Override it with `--ocr-url`; `VDI_OCR_API_KEY` supplies optional bearer authentication. **Terminal screenshots, potentially including credentials and other sensitive material, are sent to this endpoint.** Use only a trusted service and HTTPS for a non-loopback deployment. HTTP access/truncation errors are surfaced; the tool never silently switches backends or retries a command.

## Inspect a capture before typing

Single-frame mode does not click or send keyboard input. It still captures the selected browser surface and may scroll it into view. The output PNG is created with owner-only permissions and existing files are not overwritten.

C7
```sh
.venv/bin/vdi-terminal --cdp http://127.0.0.1:9222 \
  --target "$VDI_TARGET" --selector canvas --crop 20,60,1000,600 \
  --ocr rapidocr --ready --once --screenshot terminal-capture.png
```

Crop numbers are an example, not detected coordinates. Omit `--ready` if you want the setup prompt first. `--headless` is available for an already authenticated dedicated profile.

## Runtime behavior and limits

**F1 — Freshness.** Only one OCR request runs at a time. Identical screenshots skip inference; intermediate frames are not queued. Keyboard forwarding remains independent of OCR. The status line shows capture age, inference time, and queued input. Screens redraw as snapshots, without guessed scrollback deduplication. Long command output should be retained and paged inside the VM.

**F2 — Failure and exit.** Browser/OCR/input failures stop the local session and restore the local terminal. Already sent commands are not undone; pending keyboard input can be discarded on exit. Completion and remote exit status cannot be inferred reliably from pixels. Inspect the remote terminal before reconnecting or reissuing commands. A valid frame with no recognized text displays an empty screen, not evidence that a command succeeded.

**F3 — Not a complete terminal protocol.** There is no exact cursor position, remote resize protocol, file transfer, command completion detection, or reconstructed stdout/stderr/exit code. Password input is forwarded but intentionally invisible when the remote program disables echo. Full-screen programs may accept keys, but OCR does not guarantee faithful layout or refresh speed. OCR text is sanitized before local display so control characters cannot execute terminal escape sequences.

## Development

C8
```sh
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The focused regressions cover OCR protocol failure boundaries, layout-tag removal, blank frames, spatial text reconstruction, and terminal-control sanitization. Live validation against your VDI and Unlimited-OCR server remains necessary: unit tests do not establish model accuracy or client-specific keyboard/clipboard behavior.

## Validation performed

The actual CLI was exercised through a local PTY against Chromium displaying an xterm.js terminal connected to a real Bash PTY. Keyboard input produced shell-evaluated numeric results that were recovered from screenshots by RapidOCR and displayed locally. Backspace editing, interrupting a 30-second sleep with Ctrl+C, exiting with Ctrl+], and restoring local terminal settings passed. Dedicated headless browser launch, CDP attachment without closing the existing tab, and cropped single-frame capture also passed.

The 23 focused regression cases passed; the revised layout-only regression was also run separately. No claim is made of live validation against the user's VDI, noVNC clipboard integration, or an actual Unlimited-OCR server. The Unlimited-OCR protocol/parser has local regression coverage, but model latency and recognition accuracy require a running server and representative screenshots.
