# tests/interface/test_cli.py
from unittest.mock import MagicMock, patch
from interface.cli import format_response, create_app

def test_format_response_plain():
    assert format_response("你好") == "你好"

def test_format_response_strips_whitespace():
    assert format_response("  你好  ") == "你好"

def test_create_app_returns_callable():
    mock_agent = MagicMock()
    app = create_app(mock_agent)
    assert callable(app)