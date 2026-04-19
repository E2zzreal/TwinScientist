# agent/config.py
import os
import yaml

def load_config(config_path: str) -> dict:
    """Load YAML config and merge environment variables."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Merge API key from environment based on provider
    provider = config.get("provider", "anthropic")
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        config["api_key"] = api_key
    return config