from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import httpx
import pytest

from vdi_terminal.ocr import (
    OCRError,
    OCRRequestError,
    OCRResponseError,
    OCRTruncatedError,
    RapidOCROCR,
    UnlimitedOCR,
    create_ocr,
    _parse_detections,
    _reconstruct_layout,
)


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _response(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ]
    }


_PNG = b"\x89PNG\r\n\x1a\nfixture"


def test_unlimited_removes_layout_annotation_without_inserting_its_label() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raw = (
            "<|ref|>text<|/ref|>"
            "<|det|>[[0,0,100,100]]<|/det|>\n"
            "$ pwd\n/home/user"
        )
        return httpx.Response(200, json=_response(raw))

    ocr = UnlimitedOCR(
        base_url="http://127.0.0.1:8000/v1/",
        model="baidu/Unlimited-OCR",
        api_key="secret",
        timeout=30,
        max_tokens=4096,
        _transport=httpx.MockTransport(handle),
    )
    try:
        result = _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())

    assert result == "\n$ pwd\n/home/user"


def test_unlimited_does_not_remove_unpaired_grounding_text() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response("before <|ref|>literal without closing tag after"),
        )

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        assert _run(ocr.recognize(_PNG)) == (
            "before <|ref|>literal without closing tag after"
        )
    finally:
        _run(ocr.close())


def test_unlimited_unwraps_standalone_ref_and_removes_only_suffix_stop() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raw = (
            " <|ref|>status<|/ref|>\n"
            "$ echo <｜end▁of▁sentence｜> literal\n"
            "ok<｜end▁of▁sentence｜> \n"
        )
        return httpx.Response(200, json=_response(raw))

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        assert _run(ocr.recognize(_PNG)) == (
            " status\n$ echo <｜end▁of▁sentence｜> literal\nok \n"
        )
    finally:
        _run(ocr.close())


def test_unlimited_rejects_http_error_without_echoing_untrusted_body() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="\x1b[31mhostile server body\x1b[0m")

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(OCRRequestError) as raised:
            _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())
    assert "hostile" not in str(raised.value)
    assert "\x1b" not in str(raised.value)


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": {"content": ["not", "text"]}}]},
    ],
)
def test_unlimited_rejects_malformed_responses(body: Any) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(OCRResponseError):
            _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n\t",
        "<|det|>[[1, 2, 3, 4]]<|/det|>",
        " \n<|det|>[[1, 2, 3, 4]]<|/det|>\t",
        (
            "<|ref|>text<|/ref|>"
            "<|det|>[[0,0,100,100]]<|/det|>"
        ),
    ],
)
def test_unlimited_returns_empty_for_a_valid_blank_frame(content: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response(content))

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        assert _run(ocr.recognize(_PNG)) == ""
    finally:
        _run(ocr.close())


def test_unlimited_rejects_invalid_json() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(OCRResponseError):
            _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())


def test_unlimited_reports_length_truncation_instead_of_partial_success() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response("apparently valid but incomplete", finish_reason="length"),
        )

    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=10,
        _transport=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(OCRTruncatedError):
            _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())


def _box(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ]


def test_rapidocr_reconstructs_rows_indentation_and_reading_order() -> None:
    detections = _parse_detections((
        [
            [_box(30, 30, 70, 40), "file", 0.99],
            [_box(30, 10, 50, 20), "ls", 0.99],
            [_box(10, 10, 20, 20), "$", 0.99],
        ],
        [0.01, 0.02, 0.03],
    ))
    assert _reconstruct_layout(detections) == "$ ls\n  file"


def test_rapidocr_returns_empty_for_no_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyRapidOCR:
        def __init__(self, **options: Any) -> None:
            pass

        def __call__(self, png: bytes) -> tuple[None, None]:
            return None, None

    fake_module = types.ModuleType("rapidocr_onnxruntime")
    fake_module.RapidOCR = EmptyRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)

    ocr = RapidOCROCR()
    try:
        assert _run(ocr.recognize(_PNG)) == ""
    finally:
        _run(ocr.close())


def test_rapidocr_rejects_invalid_result_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedRapidOCR:
        def __init__(self, **options: Any) -> None:
            pass

        def __call__(self, png: bytes) -> dict[str, str]:
            return {"unexpected": "object"}

    fake_module = types.ModuleType("rapidocr_onnxruntime")
    fake_module.RapidOCR = MalformedRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)

    ocr = RapidOCROCR()
    try:
        with pytest.raises(OCRResponseError):
            _run(ocr.recognize(_PNG))
    finally:
        _run(ocr.close())


def test_create_ocr_never_falls_back_for_unknown_backend() -> None:
    with pytest.raises(ValueError):
        create_ocr(
            "typo",
            base_url="http://localhost:8000",
            model="baidu/Unlimited-OCR",
            api_key=None,
            timeout=30,
            max_tokens=4096,
        )


def test_closed_backend_fails_clearly() -> None:
    ocr = UnlimitedOCR(
        base_url="http://localhost:8000",
        model="baidu/Unlimited-OCR",
        api_key=None,
        timeout=1,
        max_tokens=100,
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_response("text"))
        ),
    )
    _run(ocr.close())
    with pytest.raises(OCRError):
        _run(ocr.recognize(_PNG))
