# agent/tokens.py
import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (close to Claude's tokenizer)."""
    if not text:
        return 0
    return len(_encoder.encode(text))