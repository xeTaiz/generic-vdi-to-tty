"""OCR backends for terminal screenshots.

Unlimited-OCR is a document-to-Markdown model, not an exact terminal-screen
parser.  Its output is therefore a best-effort screen snapshot: spacing and
terminal cell boundaries can be imperfect.  The optional RapidOCR backend uses
bounding-box geometry to reconstruct a monospace-like layout, but has the same
fundamental limitation.
"""

from __future__ import annotations

import asyncio
import base64
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class OCRError(RuntimeError):
    """Base class for OCR failures that are safe to present to a user."""


class OCRRequestError(OCRError):
    """The remote OCR service could not complete a request."""


class OCRResponseError(OCRError):
    """The OCR service returned an unusable response."""


class OCRTruncatedError(OCRResponseError):
    """The configured output limit cut off OCR generation."""


class OCRBackend(Protocol):
    async def recognize(self, png: bytes) -> str:
        """Recognize one PNG screenshot."""

    async def close(self) -> None:
        """Release resources owned by the backend."""


_REF_DET_TAG = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>",
    re.DOTALL,
)
_REF_TAG = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_DET_TAG = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_STOP_SUFFIX = re.compile(r"<｜end▁of▁sentence｜>(?=\s*\Z)")


def _clean_unlimited_output(raw: str) -> str:
    """Remove only Unlimited-OCR's paired structural tokens.

    Adjacent ref/det pairs are layout annotations (label plus coordinates), so
    both payloads are removed. A standalone reference is conservatively
    unwrapped and a standalone detection is removed. Unmatched tags and all
    non-token whitespace remain untouched. The model's stop token is removed
    only when it is a suffix, while any whitespace after it is retained.
    """

    cleaned = _REF_DET_TAG.sub("", raw)
    cleaned = _REF_TAG.sub(lambda match: match.group(1), cleaned)
    cleaned = _DET_TAG.sub("", cleaned)
    return _STOP_SUFFIX.sub("", cleaned)


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized:
        raise ValueError("Unlimited-OCR base URL must not be empty")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


class UnlimitedOCR:
    """Client for the official Unlimited-OCR vLLM serving recipe."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
        max_tokens: int,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError("Unlimited-OCR model name must not be empty")
        if timeout <= 0:
            raise ValueError("OCR timeout must be greater than zero")
        if max_tokens <= 0:
            raise ValueError("OCR max_tokens must be greater than zero")

        self._url = _chat_completions_url(base_url)
        self._model = model
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=_transport,
        )
        self._closed = False

    async def recognize(self, png: bytes) -> str:
        if self._closed:
            raise OCRError("Unlimited-OCR backend is closed")
        if not png:
            raise OCRError("Cannot OCR an empty PNG screenshot")

        encoded = base64.b64encode(png).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>document parsing."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        }

        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OCRRequestError(
                f"Unlimited-OCR request failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OCRRequestError(
                f"Unlimited-OCR request failed: {type(exc).__name__}"
            ) from exc

        try:
            body = response.json()
        except (ValueError, UnicodeError) as exc:
            raise OCRResponseError("Unlimited-OCR returned invalid JSON") from exc

        if not isinstance(body, dict):
            raise OCRResponseError("Unlimited-OCR response must be a JSON object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OCRResponseError("Unlimited-OCR response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OCRResponseError("Unlimited-OCR returned a malformed choice")
        if choice.get("finish_reason") == "length":
            raise OCRTruncatedError(
                "Unlimited-OCR output reached max_tokens and was truncated; "
                "increase the OCR token limit"
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OCRResponseError("Unlimited-OCR choice has no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise OCRResponseError("Unlimited-OCR message content is not text")

        cleaned = _clean_unlimited_output(content)
        return cleaned if cleaned.strip() else ""

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()


@dataclass(frozen=True)
class _Detection:
    left: float
    right: float
    top: float
    bottom: float
    text: str
    columns: int

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class _Row:
    detections: list[_Detection]
    top: float
    bottom: float

    @property
    def center_y(self) -> float:
        return statistics.fmean(item.center_y for item in self.detections)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def add(self, item: _Detection) -> None:
        self.detections.append(item)
        self.top = min(self.top, item.top)
        self.bottom = max(self.bottom, item.bottom)


def _display_columns(text: str) -> int:
    columns = 0
    for character in text:
        if character in "\r\n":
            continue
        if unicodedata.combining(character):
            continue
        columns += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return columns


def _geometry(box: Any) -> tuple[float, float, float, float]:
    try:
        points = [(float(point[0]), float(point[1])) for point in box]
    except (TypeError, ValueError, IndexError) as exc:
        raise OCRResponseError("RapidOCR returned an invalid text box") from exc
    if len(points) < 4 or not all(
        math.isfinite(coordinate) for point in points for coordinate in point
    ):
        raise OCRResponseError("RapidOCR returned an invalid text box")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    if right <= left or bottom <= top:
        raise OCRResponseError("RapidOCR returned a zero-sized text box")
    return left, right, top, bottom


def _parse_detections(raw_result: Any) -> list[_Detection]:
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise OCRResponseError("RapidOCR returned an unexpected result structure")
    items = raw_result[0]
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        raise OCRResponseError("RapidOCR returned an unexpected result structure")

    detections: list[_Detection] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise OCRResponseError("RapidOCR returned a malformed text detection")
        text = item[1]
        if not isinstance(text, str):
            raise OCRResponseError("RapidOCR returned non-text content")
        if not text:
            continue
        columns = _display_columns(text)
        if columns == 0:
            continue
        left, right, top, bottom = _geometry(item[0])
        detections.append(
            _Detection(left, right, top, bottom, text, columns)
        )

    return detections


def _same_row(row: _Row, item: _Detection) -> bool:
    overlap = min(row.bottom, item.bottom) - max(row.top, item.top)
    minimum_height = min(row.height, item.height)
    if overlap >= minimum_height * 0.3:
        return True
    return abs(row.center_y - item.center_y) <= max(row.height, item.height) * 0.45


def _cluster_rows(detections: list[_Detection]) -> list[_Row]:
    rows: list[_Row] = []
    for item in sorted(detections, key=lambda value: (value.center_y, value.left)):
        if rows and _same_row(rows[-1], item):
            rows[-1].add(item)
        else:
            rows.append(_Row([item], item.top, item.bottom))
    return rows


def _nearest_cell(value: float) -> int:
    return int(math.floor(value + 0.5))


def _reconstruct_layout(detections: list[_Detection]) -> str:
    rows = _cluster_rows(detections)
    character_widths = [
        (item.right - item.left) / item.columns
        for item in detections
        if item.columns > 0
    ]
    character_width = statistics.median(character_widths)
    if not math.isfinite(character_width) or character_width <= 0:
        raise OCRResponseError("RapidOCR returned unusable text geometry")

    left_origin = min(item.left for item in detections)
    rendered_rows: list[str] = []
    for row in rows:
        current_column = 0
        pieces: list[str] = []
        for item in sorted(row.detections, key=lambda value: value.left):
            target_column = max(
                0, _nearest_cell((item.left - left_origin) / character_width)
            )
            if target_column > current_column:
                pieces.append(" " * (target_column - current_column))
            pieces.append(item.text)
            current_column = max(current_column, target_column + item.columns)
        rendered_rows.append("".join(pieces))

    if len(rows) == 1:
        return rendered_rows[0]

    median_height = statistics.median(row.height for row in rows)
    center_gaps = [
        rows[index].center_y - rows[index - 1].center_y
        for index in range(1, len(rows))
    ]
    ordinary_gaps = [
        gap for gap in center_gaps if median_height * 0.6 <= gap <= median_height * 2.2
    ]
    line_pitch = (
        statistics.median(ordinary_gaps)
        if ordinary_gaps
        else median_height * 1.35
    )

    output = rendered_rows[0]
    for index, text in enumerate(rendered_rows[1:], start=1):
        line_breaks = max(1, _nearest_cell(center_gaps[index - 1] / line_pitch))
        output += "\n" * line_breaks + text
    return output


class RapidOCROCR:
    """CPU OCR using the models bundled by ``rapidocr_onnxruntime``."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OCRError(
                "RapidOCR backend is not installed; install the rapidocr optional "
                "dependency (rapidocr_onnxruntime)"
            ) from exc
        try:
            self._engine = RapidOCR(
                intra_op_num_threads=2,
                inter_op_num_threads=1,
            )
        except Exception as exc:
            raise OCRError(f"RapidOCR initialization failed: {exc}") from exc
        self._closed = False

    def _recognize_sync(self, png: bytes) -> str:
        try:
            # The bytes loader decodes PNG channels and converts RGB/RGBA to
            # the BGR input expected by RapidOCR's OpenCV-based pipeline.
            raw_result = self._engine(png)
            detections = _parse_detections(raw_result)
            if not detections:
                return ""
            return _reconstruct_layout(detections)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"RapidOCR inference failed: {exc}") from exc

    async def recognize(self, png: bytes) -> str:
        if self._closed:
            raise OCRError("RapidOCR backend is closed")
        if not png:
            raise OCRError("Cannot OCR an empty PNG screenshot")
        return await asyncio.to_thread(self._recognize_sync, png)

    async def close(self) -> None:
        self._closed = True


def create_ocr(
    backend: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    max_tokens: int,
) -> OCRBackend:
    """Create one reusable OCR backend instance.

    RapidOCR intentionally ignores remote-service settings. It is selected only
    when explicitly requested and is never an automatic fallback.
    """

    if backend == "unlimited":
        return UnlimitedOCR(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    if backend == "rapidocr":
        return RapidOCROCR()
    raise ValueError(
        f"Unknown OCR backend {backend!r}; expected 'unlimited' or 'rapidocr'"
    )


__all__ = [
    "OCRBackend",
    "OCRError",
    "OCRRequestError",
    "OCRResponseError",
    "OCRTruncatedError",
    "RapidOCROCR",
    "UnlimitedOCR",
    "create_ocr",
]
