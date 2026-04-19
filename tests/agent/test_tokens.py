# tests/agent/test_tokens.py
from agent.tokens import count_tokens

def test_count_tokens_english():
    text = "Hello, world!"
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0

def test_count_tokens_chinese():
    text = "你好世界"
    count = count_tokens(text)
    assert count > 0

def test_count_tokens_empty():
    assert count_tokens("") == 0