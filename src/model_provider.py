from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    value = value.lower().strip()
    if value in ["anthorpic", "anthropic"]:
        return "anthropic"
    if value in ["google", "gemini"]:
        return "gemini"
    if value in ["open-router", "openrouter"]:
        return "openrouter"
    return value


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)
    
    if provider == "openai":
        # pyrefly: ignore [missing-import]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "custom":
        # pyrefly: ignore [missing-import]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "sk-custom",
            base_url=config.base_url
        )
    elif provider == "gemini":
        # pyrefly: ignore [missing-import]
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "anthropic":
        # pyrefly: ignore [missing-import]
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "ollama":
        # pyrefly: ignore [missing-import]
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url
        )
    elif provider == "openrouter":
        # pyrefly: ignore [missing-import]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
