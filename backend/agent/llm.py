"""
LLM factory — Groq preferred, Gemini second, Ollama fallback.
Circuit breaker skips recently-failed providers automatically.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_circuit_state = {}
CIRCUIT_COOLDOWN_SECONDS = 30

# Manually selected provider (set via POST /api/provider from the UI's
# model dropdown). "auto" means: use the Groq -> Gemini -> Ollama
# preference order below, purely based on which API keys are present.
# Any other value means: try that provider first, and only fall back to
# the rest of the auto order if it errors (invalid/missing key, provider
# down, etc.) — so picking a provider in the UI never makes the app
# harder-fail than before, it just changes *which one goes first*.
_selected_provider = "auto"
_VALID_PROVIDERS = {"auto", "groq", "gemini", "ollama"}


def set_provider(provider: str) -> None:
    """Manually select which provider get_llm() should try first."""
    global _selected_provider
    normalized = (provider or "auto").strip().lower()
    if normalized not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Must be one of {sorted(_VALID_PROVIDERS)}")
    _selected_provider = normalized
    logger.info(f"[llm] provider manually set to '{_selected_provider}'")


def get_selected_provider() -> str:
    """Currently selected provider preference ('auto' or a specific provider)."""
    return _selected_provider


def _is_circuit_open(provider: str) -> bool:
    last_fail = _circuit_state.get(provider)
    if last_fail is None:
        return False
    return (time.time() - last_fail) < CIRCUIT_COOLDOWN_SECONDS


def _trip_circuit(provider: str):
    _circuit_state[provider] = time.time()
    logger.warning(f"[llm] circuit opened for '{provider}' — cooling down {CIRCUIT_COOLDOWN_SECONDS}s")


def _build_groq(temperature: float):
    from langchain_groq import ChatGroq
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def _build_gemini(temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def _build_ollama(temperature: float):
    from langchain_ollama import ChatOllama
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def get_llm(temperature: float = 0.0):
    """
    Provider preference: whichever provider is manually selected (see
    set_provider()) is tried first; "auto" (the default) uses
    Groq -> Gemini -> Ollama, based on which API keys are present.
    Circuit breaker skips recently-failed providers either way.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    auto_order = []
    if groq_key:
        auto_order.append("groq")
    if gemini_key:
        auto_order.append("gemini")
    auto_order.append("ollama")

    requested = _selected_provider
    if requested == "auto":
        providers = auto_order
    else:
        # Try the manually selected provider first, then fall back to the
        # rest of the auto-order so an unavailable manual choice degrades
        # gracefully instead of hard-failing the whole request.
        providers = [requested] + [p for p in auto_order if p != requested]

    last_error = None
    for provider in providers:
        if _is_circuit_open(provider):
            logger.info(f"[llm] skipping '{provider}' — circuit open")
            continue
        try:
            if provider == "groq":
                client = _build_groq(temperature)
            elif provider == "gemini":
                client = _build_gemini(temperature)
            else:
                client = _build_ollama(temperature)
            logger.info(f"[llm] using provider: {provider}")
            return client
        except Exception as exc:
            last_error = exc
            _trip_circuit(provider)
            logger.warning(f"[llm] provider '{provider}' failed: {exc}")

    try:
        return _build_ollama(temperature)
    except Exception:
        raise RuntimeError(
            f"All LLM providers unavailable. Last error: {last_error}. "
            "Check that Ollama is running or GROQ_API_KEY is set."
        )