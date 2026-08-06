#!/usr/bin/env python3
"""Sobe a API FastAPI (local). Uso: python scripts/run_api.py"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        "llama_qiskit_agents.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("UVICORN_RELOAD", "").lower() in ("1", "true", "yes"),
    )
