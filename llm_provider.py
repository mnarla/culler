"""Local LLM inference wrapper using llama-cpp-python.

Loads a GGUF model specified in config.py and exposes a single predict()
function that returns the raw completion text plus timing info so every
per-track decision can be benchmarked during model evaluation.

Usage:
    from llm_provider import predict

    result = predict(prompt="Is this a banger? Answer yes/no.")
    print(result["text"], result["elapsed_sec"])

Smoke test (sanity-check a freshly downloaded model):
    python llm_provider.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Config ───────────────────────────────────────────────────────────────────
try:
    import config as _cfg

    _MODEL_PATH: Path = _cfg.ACTIVE_MODEL_PATH
    _N_THREADS: int = _cfg.N_THREADS
    _N_CTX: int = _cfg.N_CTX
    _DEFAULT_MAX_TOKENS: int = _cfg.DEFAULT_MAX_TOKENS
    _TEMPERATURE: float = _cfg.TEMPERATURE
except ImportError:
    # Fallback defaults if config.py is missing (should not happen in practice)
    _MODEL_PATH = Path("models/Qwen3-8B-Q4_K_M.gguf")
    _N_THREADS = 8
    _N_CTX = 4096
    _DEFAULT_MAX_TOKENS = 200
    _TEMPERATURE = 0.0

# ── Module-level singleton ────────────────────────────────────────────────────
# Loaded lazily on first call to predict() so import doesn't block.
_llm: Optional[Any] = None
_loaded_model_path: Optional[Path] = None


def _load_model(model_path: Path) -> Any:
    """Load and return a llama_cpp.Llama instance, with clear failure messages."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "[llm_provider] ERROR: llama-cpp-python is not installed.\n"
            "  Install it with:\n"
            "    pip install llama-cpp-python\n"
            "  For CPU-only (no BLAS acceleration):\n"
            "    CMAKE_ARGS='-DGGML_BLAS=OFF' pip install llama-cpp-python --force-reinstall",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_path.exists():
        print(
            f"[llm_provider] ERROR: Model file not found at '{model_path}'.\n"
            f"  Download a GGUF and place it there, or update ACTIVE_MODEL_PATH in config.py.\n"
            f"  Candidates:\n"
            f"    models/phi-4-mini-instruct-Q4_K_M.gguf\n"
            f"    models/Qwen3-4B-Q4_K_M.gguf\n"
            f"    models/Qwen3-8B-Q4_K_M.gguf\n"
            f"    models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[llm_provider] Loading model: {model_path} ...", flush=True)
    load_start = time.perf_counter()

    llm = Llama(
        model_path=str(model_path),
        n_threads=_N_THREADS,
        n_ctx=_N_CTX,
        # Let llama-cpp-python auto-detect the chat template from GGUF metadata.
        # This means Phi-4-mini, Qwen3, and Mistral all work without any
        # model-specific template code here.
        chat_format=None,
        verbose=False,
    )

    elapsed = time.perf_counter() - load_start
    print(f"[llm_provider] Model loaded in {elapsed:.2f}s  (threads={_N_THREADS}, ctx={_N_CTX})", flush=True)
    return llm


def get_model(force_reload: bool = False) -> Any:
    """Return the loaded model singleton, loading it on first call.

    Args:
        force_reload: If True, reload the model even if already loaded
                      (useful when ACTIVE_MODEL_PATH changed between calls).
    """
    global _llm, _loaded_model_path

    if force_reload or _llm is None or _loaded_model_path != _MODEL_PATH:
        _llm = _load_model(_MODEL_PATH)
        _loaded_model_path = _MODEL_PATH

    return _llm


def predict(
    prompt: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _TEMPERATURE,
    stop: Optional[list] = None,
) -> Dict[str, Any]:
    """Run a single completion and return text + timing metadata.

    Args:
        prompt:      The full prompt string (raw, not chat-formatted).
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).
        stop:        Optional list of stop sequences.

    Returns:
        {
            "text":         str   — raw completion text (stripped),
            "prompt_tokens": int  — tokens in the prompt,
            "completion_tokens": int,
            "total_tokens": int,
            "start_time":   float — unix timestamp,
            "end_time":     float — unix timestamp,
            "elapsed_sec":  float — wall-clock seconds for this call,
        }
    """
    llm = get_model()

    start_time = time.time()
    perf_start = time.perf_counter()

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [],
        echo=False,
    )

    perf_end = time.perf_counter()
    end_time = time.time()

    text = response["choices"][0]["text"].strip()
    usage = response.get("usage", {})

    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": round(perf_end - perf_start, 3),
    }


def predict_chat(
    messages: list,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _TEMPERATURE,
    stop: Optional[list] = None,
) -> Dict[str, Any]:
    """Run a chat-formatted completion using the model's native template.

    Uses llama-cpp-python's create_chat_completion(), which reads the chat
    template embedded in the GGUF metadata — no manual prompt assembly needed.

    Args:
        messages: OpenAI-style list of {"role": ..., "content": ...} dicts.
        max_tokens, temperature, stop: Same as predict().

    Returns:
        Same dict shape as predict(), with "text" = assistant reply content.
    """
    llm = get_model()

    start_time = time.time()
    perf_start = time.perf_counter()

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [],
    )

    perf_end = time.perf_counter()
    end_time = time.time()

    text = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage", {})

    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": round(perf_end - perf_start, 3),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("llm_provider.py — model smoke test")
    print("=" * 60)
    print(f"  Model path : {_MODEL_PATH}")
    print(f"  Threads    : {_N_THREADS}")
    print(f"  Context    : {_N_CTX} tokens")
    print()

    SMOKE_PROMPT = (
        "You are a music taste classifier. Answer in one word.\n"
        "Should a track by a heavily repeated artist in a playlist "
        "that was added 2000 days ago be kept or skipped?\n"
        "Answer: "
    )

    print(f"Prompt: {SMOKE_PROMPT.strip()}\n")

    result = predict(SMOKE_PROMPT, max_tokens=20, temperature=0.0)

    print(f"Response     : {result['text']!r}")
    print(f"Elapsed      : {result['elapsed_sec']}s")
    print(f"Tokens used  : {result['total_tokens']} "
          f"(prompt={result['prompt_tokens']}, completion={result['completion_tokens']})")
    print()
    print("[OK] Smoke test complete — model is working.")
    print("     Tokens/sec ≈", round(result["completion_tokens"] / result["elapsed_sec"], 1),
          "tok/s" if result["elapsed_sec"] > 0 else "(instant)")
