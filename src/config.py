from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    import os
    from dotenv import load_dotenv

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = os.getenv("LLM_PROVIDER", "gemini")
    model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    api_key = None
    base_url = None
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

    judge_provider = os.getenv("JUDGE_LLM_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_LLM_MODEL", model_name)
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=judge_model
    )
