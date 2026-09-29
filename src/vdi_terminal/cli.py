"""Local launcher for the browser/OCR bridge."""
from __future__ import annotations

import argparse
import asyncio
import math
import os
from pathlib import Path
import sys

from .browser import BrowserSession
from .ocr import create_ocr
from .terminal import TerminalMirror, safe_screen


def crop_value(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height in pixels") from exc
    if len(parts) != 4 or min(parts[:2]) < 0 or min(parts[2:]) <= 0:
        raise argparse.ArgumentTypeError("crop requires nonnegative x,y and positive width,height")
    return parts


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Mirror a browser VDI terminal locally using OCR; Ctrl+] exits locally.",
        epilog="OCR is approximate, not an SSH byte stream. No software or network access is required inside the VM.",
    )
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Open a VDI URL in a dedicated persistent Chromium profile")
    source.add_argument("--cdp", dest="cdp_url", help="Attach to an existing Chromium CDP endpoint (keep it loopback-only)")
    result.add_argument("--target", help="URL/title substring selecting exactly one tab when attaching with --cdp")
    result.add_argument("--profile", default=str(Path.home() / ".local/share/vdi-terminal/browser"), help="Dedicated browser profile directory; contains login credentials")
    result.add_argument("--selector", default="canvas", help="Unique remote-display CSS selector (default: canvas)")
    result.add_argument("--frame", help="CSS selector for an iframe containing the remote display")
    result.add_argument("--crop", type=crop_value, help="Terminal rectangle x,y,width,height relative to the display screenshot")
    result.add_argument("--rfb-expression", help="Trusted JavaScript expression for the existing noVNC RFB object; enables its clipboard and keyboard API")
    result.add_argument("--ocr", choices=("unlimited", "rapidocr"), default="unlimited", help="OCR backend; rapidocr requires the optional cpu extra")
    result.add_argument("--ocr-url", default="http://127.0.0.1:8000/v1", help="Unlimited-OCR vLLM base URL; screenshots are sent here")
    result.add_argument("--ocr-model", default="baidu/Unlimited-OCR")
    result.add_argument("--ocr-api-key-env", default="VDI_OCR_API_KEY", help="Environment variable containing OCR server API key")
    result.add_argument("--ocr-timeout", type=positive_float, default=60.0)
    result.add_argument("--max-tokens", type=positive_int, default=8192)
    result.add_argument("--interval", type=positive_float, default=0.5, help="Minimum seconds between captures; OCR requests never overlap")
    result.add_argument("--ready", action="store_true", help="Skip the local readiness prompt; only use for an already prepared terminal")
    result.add_argument("--once", action="store_true", help="Recognize one screen and print it without sending keyboard or focus events")
    result.add_argument("--screenshot", type=Path, help="Save the captured PNG in --once mode (may contain sensitive data)")
    result.add_argument("--headless", action="store_true", help="Launch without a window; requires an already authenticated profile")
    return result


async def run(args):
    browser = BrowserSession(
        url=args.url, cdp_url=args.cdp_url, target=args.target, profile=args.profile,
        selector=args.selector, frame=args.frame, crop=args.crop,
        rfb_expression=args.rfb_expression, headless=args.headless,
    )
    ocr = None
    try:
        await browser.open()
        if not args.ready:
            print("In the browser: log in, open the VM terminal, enlarge its font, and keep it visible.", file=sys.stderr)
            print("Use --crop to exclude desktop chrome. Do not resize the VDI while attached.", file=sys.stderr)
            await asyncio.to_thread(input, "Press Enter here when the remote terminal is ready: ")
        ocr = create_ocr(
            args.ocr, base_url=args.ocr_url, model=args.ocr_model,
            api_key=os.environ.get(args.ocr_api_key_env), timeout=args.ocr_timeout,
            max_tokens=args.max_tokens,
        )
        if args.once:
            png = await browser.screenshot()
            if args.screenshot:
                # Refuse to overwrite an existing file and restrict new screenshots.
                fd = os.open(args.screenshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as image:
                    image.write(png)
            print(safe_screen(await ocr.recognize(png)))
            return
        await browser.focus()
        await TerminalMirror(browser, ocr, interval=args.interval).run()
    finally:
        try:
            if ocr is not None:
                await ocr.close()
        finally:
            await browser.close()


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.target and not args.cdp_url:
        argument_parser.error("--target requires --cdp")
    if args.screenshot and not args.once:
        argument_parser.error("--screenshot requires --once")
    if not args.once and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        argument_parser.error("interactive mode requires a local terminal; use --once for a single OCR capture")
    if not args.ready and not sys.stdin.isatty():
        argument_parser.error("noninteractive capture requires --ready")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"vdi-terminal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
