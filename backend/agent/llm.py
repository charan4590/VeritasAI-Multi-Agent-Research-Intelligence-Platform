"""
LLM factory — Groq preferred for deployment, Ollama for local dev.

#3: Fallback chain + circuit breaker.
If the primary provider fails (Ollama unreachable, Groq rate-limited),
automatically fall back to the next provider instead of crashing the
whole agent run. A circuit breaker remembers recent failures for a
provider and skips it for a cooldown window rather than retrying a
known-dead service on every single call.
"""
import os
import time
import logging

logger = logging.getLogger(__name__)

# Circuit breaker state: provider_name -> last_failure_timestamp
_circuit_state = {}
CIRCUIT_COOLDOWN_SECONDS = 30


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
    Returns an LLM client following provider preference + circuit breaker.
    Preference: Groq first if configured (faster, no local dependency),
    else Ollama. If the preferred provider's circuit is open (recently
    failed), the other provider is tried instead.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    preferred = "groq" if groq_key else "ollama"
    fallback = "ollama" if preferred == "groq" else "groq"

    providers = [preferred, fallback]
    last_error = None

    for provider in providers:
        if provider == "groq" and not groq_key:
            continue
        if _is_circuit_open(provider):
            logger.info(f"[llm] skipping '{provider}' — circuit open")
            continue
        try:
            if provider == "groq":
                client = _build_groq(temperature)
            else:
                client = _build_ollama(temperature)
            logger.info(f"[llm] using provider: {provider}")
            return client
        except Exception as exc:
            last_error = exc
            _trip_circuit(provider)
            logger.warning(f"[llm] provider '{provider}' failed to initialize: {exc}")

    # Both unavailable — last resort, try Ollama anyway even if circuit open
    # (better to attempt and fail with a clear error than refuse silently)
    try:
        return _build_ollama(temperature)
    except Exception:
        raise RuntimeError(
            f"All LLM providers unavailable. Last error: {last_error}. "
            "Check that Ollama is running or GROQ_API_KEY is set."
        )
