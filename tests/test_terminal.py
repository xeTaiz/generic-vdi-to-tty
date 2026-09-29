from vdi_terminal.terminal import safe_screen


def test_ocr_cannot_inject_terminal_controls_or_bidi_overrides():
    assert safe_screen("ok\x1b]52;c;c2VjcmV0\x07\u202eevil\x9b31m") == "ok\ufffd]52;c;c2VjcmV0\ufffd\ufffdevil\ufffd31m"


def test_terminal_text_preserves_spacing_unicode_and_normalizes_newlines():
    assert safe_screen("  λ = 1\r\n\tprint(λ)\rnext") == "  λ = 1\n        print(λ)\nnext"
