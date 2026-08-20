"""Central configuration for the Skip-Prediction Playlist Agent (Culler).

Swap ACTIVE_MODEL_PATH to point at whichever GGUF file you want to benchmark.
All other files read from here — never hardcode a model path elsewhere.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Model selection — change this line to switch models between benchmark runs.
# ─────────────────────────────────────────────────────────────────────────────

# Candidates (download whichever you're testing into models/ and update below):
#   models/phi-4-mini-instruct-Q4_K_M.gguf          (3.8B, fastest)
#   models/qwen2.5-coder-7b-instruct-Q4_K_M.gguf    (7B, current baseline)
#   models/Qwen3-4B-Q4_K_M.gguf                     (4B)
#   models/Qwen3-8B-Q4_K_M.gguf                     (8B)
#   models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf     (7B)

ACTIVE_MODEL_PATH: Path = Path("models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")

# ─────────────────────────────────────────────────────────────────────────────
# Inference settings
# ─────────────────────────────────────────────────────────────────────────────

# Pin to physical cores — hyperthreads hurt throughput on AVX2 matrix ops
N_THREADS: int = 8

# Context window: enough for a structured 5-feature prompt + reasoning + JSON
# Anything ≥2048 is fine; 4096 gives comfortable headroom for future changes
N_CTX: int = 4096

# Max tokens for a single prediction response
DEFAULT_MAX_TOKENS: int = 200

# Temperature — 0.0 for deterministic verdicts (skip/keep)
TEMPERATURE: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH: Path = Path("skip_predictor.db")
