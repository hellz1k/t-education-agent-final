"""Build or refresh dense embeddings and BM25 index artifacts under data/index/."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.config import load_config
from t_edu_agent.retrieval.hybrid import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate cached retrieval index.")
    parser.add_argument("--force", action="store_true", help="Force rebuilding dense embeddings.")
    args = parser.parse_args()

    cfg = load_config()
    t0 = time.perf_counter()
    retriever = HybridRetriever(
        cfg.paths.processed_dir,
        cfg.paths.index_dir,
        cfg.retrieval,
        force_rebuild=args.force,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    print("=== T-Education Retrieval Index Summary ===")
    print(f"Processed dir   : {cfg.paths.processed_dir}")
    print(f"Index dir       : {cfg.paths.index_dir}")
    print(f"Backend         : {retriever.embedder.backend_name}")
    print(f"Model           : {retriever.embedder.model_name}")
    print(f"Documents       : {len(retriever.documents)}")
    print(f"Chunks          : {len(retriever.chunks)}")
    print(f"Embeddings shape: {tuple(retriever.embeddings.shape)}")
    print(f"Cache hit       : {retriever.cache_hit}")
    print(f"Elapsed         : {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
