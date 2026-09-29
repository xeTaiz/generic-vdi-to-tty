from __future__ import annotations

import asyncio

from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Frame,
    Locator,
    Page,
    Playwright,
    async_playwright,
)


class BrowserSessionError(RuntimeError):
    """An actionable failure while controlling the VDI browser session."""


_MODIFIER_KEYS = {
    "alt": (0xFFE9, "AltLeft", "Alt"),
    "control": (0xFFE3, "ControlLeft", "Control"),
    "controlormeta": (0xFFE3, "ControlLeft", "Control"),
    "meta": (0xFFEB, "MetaLeft", "Meta"),
    "shift": (0xFFE1, "ShiftLeft", "Shift"),
}
_MODIFIER_ORDER = ("Control", "Alt", "Shift", "Meta")

_SPECIAL_KEYS = {
    "backspace": (0xFF08, "Backspace"),
    "tab": (0xFF09, "Tab"),
    "enter": (0xFF0D, "Enter"),
    "return": (0xFF0D, "Enter"),
    "escape": (0xFF1B, "Escape"),
    "esc": (0xFF1B, "Escape"),
    "home": (0xFF50, "Home"),
    "arrowleft": (0xFF51, "ArrowLeft"),
    "left": (0xFF51, "ArrowLeft"),
    "arrowup": (0xFF52, "ArrowUp"),
    "up": (0xFF52, "ArrowUp"),
    "arrowright": (0xFF53, "ArrowRight"),
    "right": (0xFF53, "ArrowRight"),
    "arrowdown": (0xFF54, "ArrowDown"),
    "down": (0xFF54, "ArrowDown"),
    "pageup": (0xFF55, "PageUp"),
    "pagedown": (0xFF56, "PageDown"),
    "end": (0xFF57, "End"),
    "insert": (0xFF63, "Insert"),
    "delete": (0xFFFF, "Delete"),
    "space": (0x20, "Space"),
}
for _function_number in range(1, 25):
    _SPECIAL_KEYS[f"f{_function_number}"] = (
        0xFFBD + _function_number,
        f"F{_function_number}",
    )

_ASCII_CODES = {
    " ": "Space",
    "-": "Minus",
    "_": "Minus",
    "=": "Equal",
    "+": "Equal",
    "[": "BracketLeft",
    "{": "BracketLeft",
    "]": "BracketRight",
    "}": "BracketRight",
    "\\": "Backslash",
    "|": "Backslash",
    ";": "Semicolon",
    ":": "Semicolon",
    "'": "Quote",
    '"': "Quote",
    ",": "Comma",
    "<": "Comma",
    ".": "Period",
    ">": "Period",
    "/": "Slash",
    "?": "Slash",
    "`": "Backquote",
    "~": "Backquote",
    "!": "Digit1",
    "@": "Digit2",
    "#": "Digit3",
    "$": "Digit4",
    "%": "Digit5",
    "^": "Digit6",
    "&": "Digit7",
    "*": "Digit8",
    "(": "Digit9",
    ")": "Digit0",
}
_SHIFTED_ASCII = {
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
    "`": "~",
}


class BrowserSession:
    """Own or attach to the browser page containing an offline VDI terminal."""

    def __init__(
        self,
        *,
        url: str | None,
        cdp_url: str | None,
        target: str | None,
        profile: str,
        selector: str,
        frame: str | None,
        crop: tuple[int, int, int, int] | None,
        rfb_expression: str | None,
        headless: bool = False,
    ) -> None:
        if bool(url) == bool(cdp_url):
            raise BrowserSessionError(
                "Specify exactly one browser source: either a VDI URL or a CDP endpoint."
            )
        if not selector or not selector.strip():
            raise BrowserSessionError("The terminal CSS selector must not be empty.")
        if frame is not None and not frame.strip():
            raise BrowserSessionError("The iframe CSS selector must not be empty.")
        if cdp_url and target is not None and not target.strip():
            raise BrowserSessionError("The CDP target filter must not be empty.")
        if url and not profile.strip():
            raise BrowserSessionError(
                "A persistent profile directory is required when opening a VDI URL."
            )
        if crop is not None:
            if len(crop) != 4:
                raise BrowserSessionError("Crop must contain x, y, width, and height.")
            x, y, width, height = crop
            if any(not isinstance(value, int) for value in crop):
                raise BrowserSessionError("Crop x, y, width, and height must be integers.")
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise BrowserSessionError(
                    "Crop x and y must be non-negative, and width and height must be positive."
                )
        if rfb_expression is not None and not rfb_expression.strip():
            raise BrowserSessionError("The noVNC RFB expression must not be empty.")

        self.url = url
        self.cdp_url = cdp_url
        self.target = target
        self.profile = profile
        self.selector = selector
        self.frame_selector = frame
        self.crop = crop
        self.rfb_expression = rfb_expression
        self.headless = headless

        self.page: Page | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._owns_context = False

    async def open(self) -> None:
        """Open a dedicated persistent browser or select exactly one CDP page."""
        if self._playwright is not None:
            raise BrowserSessionError("The browser session is already open.")

        try:
            playwright = await async_playwright().start()
        except Exception as exc:
            raise BrowserSessionError(f"Could not start Playwright: {exc}") from exc
        self._playwright = playwright
        try:
            if self.url is not None:
                await self._open_persistent(playwright)
            else:
                await self._open_cdp(playwright)
        except BaseException:
            await self._cleanup_after_failed_open()
            raise

    async def _open_persistent(self, playwright: Playwright) -> None:
        profile = Path(self.profile).expanduser()
        try:
            profile.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise BrowserSessionError(
                f"Could not create persistent profile directory {str(profile)!r}: {exc}"
            ) from exc
        profile_path = str(profile)
        try:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=profile_path,
                headless=self.headless,
            )
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not launch Chromium with persistent profile {profile_path!r}: {exc}"
            ) from exc

        self._context = context
        self._owns_context = True
        pages = context.pages
        if len(pages) == 1 and pages[0].url == "about:blank":
            page = pages[0]
        else:
            try:
                page = await context.new_page()
            except Exception as exc:
                raise BrowserSessionError(
                    f"Chromium started, but a VDI tab could not be created: {exc}"
                ) from exc

        self.page = page
        try:
            await page.goto(self.url, wait_until="domcontentloaded")
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not navigate the dedicated VDI tab to {self.url!r}: {exc}"
            ) from exc

    async def _open_cdp(self, playwright: Playwright) -> None:
        assert self.cdp_url is not None
        try:
            browser = await playwright.chromium.connect_over_cdp(self.cdp_url)
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not attach to the browser at CDP endpoint {self.cdp_url!r}: {exc}"
            ) from exc

        self._browser = browser
        pages = [page for context in browser.contexts for page in context.pages]
        selected: list[Page]
        if self.target is None:
            selected = pages
        else:
            needle = self.target.casefold()
            selected = []
            for page in pages:
                try:
                    title = await page.title()
                except Exception:
                    title = ""
                if needle in page.url.casefold() or needle in title.casefold():
                    selected.append(page)

        if len(selected) != 1:
            candidates = await self._describe_pages(pages)
            filter_description = (
                "without a target filter"
                if self.target is None
                else f"matching URL/title filter {self.target!r}"
            )
            raise BrowserSessionError(
                f"CDP page selection {filter_description} found {len(selected)} tabs; "
                "exactly one is required. Open/close tabs or pass a more specific "
                f"--target. Available tabs: {candidates}"
            )

        self.page = selected[0]
        self._context = self.page.context
        self._owns_context = False

    async def _describe_pages(self, pages: list[Page]) -> str:
        if not pages:
            return "none"
        descriptions: list[str] = []
        for page in pages:
            try:
                title = await page.title()
            except Exception:
                title = "<title unavailable>"
            descriptions.append(f"{title!r} ({page.url})")
        return "; ".join(descriptions)

    async def _cleanup_after_failed_open(self) -> None:
        if self._owns_context and self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self.page = None
        self._browser = None
        self._context = None
        self._playwright = None
        self._owns_context = False

    def _require_page(self) -> Page:
        if self.page is None or self._playwright is None:
            raise BrowserSessionError("The browser session is not open.")
        if self.page.is_closed():
            raise BrowserSessionError("The selected VDI tab has been closed.")
        return self.page

    async def _resolve_frame(self) -> Frame:
        page = self._require_page()
        if self.frame_selector is None:
            return page.main_frame

        frame_locator = await self._unique_visible_locator(
            page.locator(self.frame_selector),
            f"iframe selector {self.frame_selector!r}",
        )
        try:
            element = await frame_locator.element_handle()
            if element is None:
                raise BrowserSessionError(
                    f"The selected iframe {self.frame_selector!r} detached before use."
                )
            frame = await element.content_frame()
        except BrowserSessionError:
            raise
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not enter iframe {self.frame_selector!r}: {exc}"
            ) from exc
        if frame is None:
            raise BrowserSessionError(
                f"Selector {self.frame_selector!r} is visible but is not a loaded iframe."
            )
        return frame

    async def _terminal_locator(self) -> Locator:
        frame = await self._resolve_frame()
        return await self._unique_visible_locator(
            frame.locator(self.selector),
            f"terminal selector {self.selector!r}",
        )

    async def _unique_visible_locator(self, locator: Locator, description: str) -> Locator:
        try:
            count = await locator.count()
            visible: list[Locator] = []
            for index in range(count):
                candidate = locator.nth(index)
                if await candidate.is_visible():
                    visible.append(candidate)
        except Exception as exc:
            raise BrowserSessionError(f"Could not inspect {description}: {exc}") from exc

        if not visible:
            raise BrowserSessionError(
                f"No visible element matches {description}. Check the selector and that "
                "the VDI page has finished loading."
            )
        if len(visible) != 1:
            raise BrowserSessionError(
                f"{description.capitalize()} matches {len(visible)} visible elements; "
                "use a selector that identifies exactly one."
            )
        return visible[0]

    async def _terminal_box(self, locator: Locator) -> dict[str, float]:
        try:
            await locator.scroll_into_view_if_needed()
            box = await locator.bounding_box()
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not measure terminal selector {self.selector!r}: {exc}"
            ) from exc
        if box is None or box["width"] <= 0 or box["height"] <= 0:
            raise BrowserSessionError(
                f"Terminal selector {self.selector!r} has no visible bounding box."
            )
        return box

    def _validate_crop_bounds(self, box: dict[str, float]) -> None:
        if self.crop is None:
            return
        x, y, width, height = self.crop
        if x + width > box["width"] or y + height > box["height"]:
            raise BrowserSessionError(
                f"Crop {self.crop!r} exceeds terminal bounds "
                f"({box['width']:.1f} x {box['height']:.1f} CSS pixels)."
            )

    async def focus(self) -> None:
        """Click the terminal (or crop) center once when the caller is ready."""
        page = self._require_page()
        locator = await self._terminal_locator()
        box = await self._terminal_box(locator)
        self._validate_crop_bounds(box)
        if self.crop is None:
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
        else:
            crop_x, crop_y, width, height = self.crop
            x = box["x"] + crop_x + width / 2
            y = box["y"] + crop_y + height / 2
        try:
            await page.mouse.click(x, y)
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not focus the VDI terminal at ({x:.1f}, {y:.1f}): {exc}"
            ) from exc

    async def screenshot(self) -> bytes:
        """Capture the terminal locator, optionally cropped in locator CSS pixels."""
        page = self._require_page()
        locator = await self._terminal_locator()
        try:
            if self.crop is None:
                return await locator.screenshot(type="png", scale="css")
            box = await self._terminal_box(locator)
            self._validate_crop_bounds(box)
            x, y, width, height = self.crop
            return await page.screenshot(
                type="png",
                scale="css",
                clip={
                    "x": box["x"] + x,
                    "y": box["y"] + y,
                    "width": width,
                    "height": height,
                },
            )
        except BrowserSessionError:
            raise
        except Exception as exc:
            raise BrowserSessionError(
                f"Could not capture terminal selector {self.selector!r}: {exc}"
            ) from exc

    async def send_text(self, text: str) -> None:
        """Type text with paced key events.

        noVNC input uses direct RFB key events. Non-Latin Unicode is encoded
        with the standard X11 ``0x01000000 | codepoint`` keysym convention.
        The RFB server and guest keyboard stack must support those keysyms.
        """
        if not isinstance(text, str):
            raise BrowserSessionError("Text input must be a string.")
        self._require_page()
        if not text:
            return
        if self.rfb_expression is not None:
            await self._run_rfb_steps(self._rfb_text_steps(text), sequential=True)
            return

        unsupported = [
            (index, character)
            for index, character in enumerate(text)
            if not (0x20 <= ord(character) <= 0x7E or character in "\n\t")
        ]
        if unsupported:
            index, character = unsupported[0]
            codepoint = f"U+{ord(character):04X}"
            raise BrowserSessionError(
                f"Keyboard text mode cannot safely type {codepoint} at character "
                f"offset {index}. Use ASCII printable text/newline/tab, or configure "
                "--rfb-expression for direct Unicode RFB input."
            )

        page = self._require_page()
        error: Exception | None = None
        release_error: Exception | None = None
        try:
            chunk: list[str] = []
            for character in text:
                if character in "\n\t":
                    if chunk:
                        await page.keyboard.type("".join(chunk), delay=5)
                        chunk.clear()
                    await page.keyboard.press(
                        "Enter" if character == "\n" else "Tab", delay=5
                    )
                else:
                    chunk.append(character)
            if chunk:
                await page.keyboard.type("".join(chunk), delay=5)
        except Exception as exc:
            error = exc
        finally:
            release_error = await self._release_keyboard_modifiers_shielded(page)
        if error is not None:
            raise BrowserSessionError(f"Could not type into the VDI terminal: {error}") from error
        if release_error is not None:
            raise BrowserSessionError(
                f"Text was typed, but browser modifier keys could not be released: {release_error}"
            ) from release_error

    async def send_paste(self, text: str) -> None:
        """Request a best-effort noVNC clipboard paste, or type without RFB."""
        if not isinstance(text, str):
            raise BrowserSessionError("Paste input must be a string.")
        self._require_page()
        if not text:
            return
        if self.rfb_expression is None:
            await self.send_text(text)
            return
        await self._run_rfb_steps(
            self._rfb_steps("Control+Shift+V"),
            clipboard=text,
        )

    async def send_key(self, key: str) -> None:
        """Send one Playwright-style key or modifier combination."""
        if not isinstance(key, str) or not key.strip():
            raise BrowserSessionError("Key input must be a non-empty Playwright key name.")
        page = self._require_page()
        if self.rfb_expression is not None:
            await self._run_rfb_steps(self._rfb_steps(key))
            return

        error: Exception | None = None
        release_error: Exception | None = None
        try:
            await page.keyboard.press(key)
        except Exception as exc:
            error = exc
        finally:
            release_error = await self._release_keyboard_modifiers_shielded(page)
        if error is not None:
            raise BrowserSessionError(
                f"Could not send Playwright key {key!r} to the VDI terminal: {error}"
            ) from error
        if release_error is not None:
            raise BrowserSessionError(
                f"Key {key!r} was sent, but browser modifier keys could not be released: "
                f"{release_error}"
            ) from release_error

    async def _release_keyboard_modifiers(self, page: Page) -> Exception | None:
        first_error: Exception | None = None
        for modifier in reversed(_MODIFIER_ORDER):
            try:
                await page.keyboard.up(modifier)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    async def _release_keyboard_modifiers_shielded(
        self, page: Page
    ) -> Exception | None:
        release_task = asyncio.create_task(self._release_keyboard_modifiers(page))
        try:
            return await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await release_task
            raise

    def _rfb_steps(self, key: str) -> list[dict[str, Any]]:
        parts = key.split("+")
        if not parts or any(not part for part in parts):
            raise BrowserSessionError(
                f"Unsupported noVNC key syntax {key!r}; use forms such as Enter, "
                "Control+C, or Control+Shift+V."
            )

        modifiers: list[tuple[int, str, str]] = []
        main: str | None = None
        for part in parts:
            modifier = _MODIFIER_KEYS.get(part.casefold())
            if modifier is not None:
                if not any(existing[2] == modifier[2] for existing in modifiers):
                    modifiers.append(modifier)
            elif main is None:
                main = part
            else:
                raise BrowserSessionError(
                    f"Unsupported noVNC key combination {key!r}: it contains more "
                    "than one non-modifier key."
                )

        steps = [
            {"keysym": keysym, "code": code, "label": label}
            for keysym, code, label in sorted(
                modifiers, key=lambda item: _MODIFIER_ORDER.index(item[2])
            )
        ]
        if main is not None:
            keysym, code = self._rfb_main_key(main, any(item[2] == "Shift" for item in modifiers))
            steps.append({"keysym": keysym, "code": code, "label": main})
        if not steps:
            raise BrowserSessionError(f"Unsupported empty noVNC key combination {key!r}.")
        return steps

    def _rfb_main_key(self, key: str, shifted: bool) -> tuple[int, str]:
        special = _SPECIAL_KEYS.get(key.casefold())
        if special is not None:
            return special
        if len(key) != 1 or not 0x20 <= ord(key) <= 0x7E:
            raise BrowserSessionError(
                f"noVNC direct input does not support key {key!r}. Use a printable "
                "ASCII key, Enter/Tab/navigation keys, or F1-F24."
            )

        character = key
        if "a" <= character <= "z":
            code = f"Key{character.upper()}"
            if shifted:
                character = character.upper()
        elif "A" <= character <= "Z":
            code = f"Key{character}"
        elif character.isdigit():
            code = f"Digit{character}"
            if shifted:
                character = _SHIFTED_ASCII[character]
        else:
            code = _ASCII_CODES.get(character)
            if code is None:
                raise BrowserSessionError(
                    f"noVNC direct input does not have a key-code mapping for {key!r}."
                )
            if shifted:
                character = _SHIFTED_ASCII.get(character, character)
        return ord(character), code

    def _rfb_text_steps(self, text: str) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for index, character in enumerate(text):
            codepoint = ord(character)
            if character == "\n":
                keysym, code = _SPECIAL_KEYS["enter"]
            elif character == "\t":
                keysym, code = _SPECIAL_KEYS["tab"]
            elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
                raise BrowserSessionError(
                    f"noVNC direct text input does not support control character "
                    f"U+{codepoint:04X} at offset {index}; only newline and tab are "
                    "supported controls."
                )
            elif 0xD800 <= codepoint <= 0xDFFF:
                raise BrowserSessionError(
                    f"noVNC direct text input cannot encode surrogate U+{codepoint:04X} "
                    f"at offset {index}."
                )
            elif codepoint <= 0x7E:
                keysym, code = self._rfb_main_key(character, shifted=False)
            elif codepoint <= 0xFF:
                keysym, code = codepoint, "Unidentified"
            else:
                keysym, code = 0x01000000 | codepoint, "Unidentified"
            steps.append({"keysym": keysym, "code": code, "label": character})
        return steps

    async def _run_rfb_steps(
        self,
        steps: list[dict[str, Any]],
        *,
        clipboard: str | None = None,
        sequential: bool = False,
    ) -> None:
        frame = await self._resolve_frame()
        assert self.rfb_expression is not None
        script = (
            "async (payload) => {\n"
            f"  const rfb = ({self.rfb_expression});\n"
            "  if (!rfb) throw new Error('the RFB expression returned no object');\n"
            "  if (typeof rfb.sendKey !== 'function') "
            "throw new Error('the selected object has no sendKey method');\n"
            "  if (payload.clipboard !== null) {\n"
            "    if (typeof rfb.clipboardPasteFrom !== 'function') "
            "throw new Error('the selected object has no clipboardPasteFrom method');\n"
            "    await Promise.resolve(rfb.clipboardPasteFrom(payload.clipboard));\n"
            "    await new Promise((resolve) => setTimeout(resolve, 75));\n"
            "  }\n"
            "  if (payload.sequential) {\n"
            "    for (const step of payload.steps) {\n"
            "      let pressed = false;\n"
            "      try {\n"
            "        rfb.sendKey(step.keysym, step.code, true);\n"
            "        pressed = true;\n"
            "        await new Promise((resolve) => setTimeout(resolve, 5));\n"
            "      } finally {\n"
            "        if (pressed) rfb.sendKey(step.keysym, step.code, false);\n"
            "      }\n"
            "    }\n"
            "    return;\n"
            "  }\n"
            "  const pressed = [];\n"
            "  let primaryError = null;\n"
            "  try {\n"
            "    for (const step of payload.steps) {\n"
            "      rfb.sendKey(step.keysym, step.code, true);\n"
            "      pressed.push(step);\n"
            "    }\n"
            "  } catch (error) { primaryError = error; }\n"
            "  let releaseError = null;\n"
            "  for (const step of pressed.reverse()) {\n"
            "    try { rfb.sendKey(step.keysym, step.code, false); }\n"
            "    catch (error) { if (releaseError === null) releaseError = error; }\n"
            "  }\n"
            "  if (primaryError !== null) throw primaryError;\n"
            "  if (releaseError !== null) throw releaseError;\n"
            "}"
        )
        try:
            await frame.evaluate(
                script,
                {
                    "steps": steps,
                    "clipboard": clipboard,
                    "sequential": sequential,
                },
            )
        except Exception as exc:
            if clipboard is not None:
                operation = "paste text through"
            elif sequential:
                operation = "type text through"
            else:
                operation = "send a key through"
            raise BrowserSessionError(
                f"Could not {operation} the configured noVNC RFB object. Check that "
                f"--rfb-expression resolves inside the selected VDI frame: {exc}"
            ) from exc

    async def close(self) -> None:
        """Release resources without navigating or closing a CDP-attached browser."""
        close_error: Exception | None = None
        if self._owns_context and self._context is not None:
            try:
                await self._context.close()
            except Exception as exc:
                close_error = exc

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                if close_error is None:
                    close_error = exc

        self.page = None
        self._browser = None
        self._context = None
        self._playwright = None
        self._owns_context = False
        if close_error is not None:
            raise BrowserSessionError(f"Could not cleanly close the browser session: {close_error}") from close_error
