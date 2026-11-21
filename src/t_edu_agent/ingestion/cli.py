"""Command-line entry point for repeatable knowledge base ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from t_edu_agent.config import load_config
from t_edu_agent.ingestion.pipeline import add_file_and_reingest, run_ingestion


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for ingestion commands."""
    parser = argparse.ArgumentParser(
        prog="t-edu-ingest",
        description="Build or update the cleaned T-Education knowledge base and source registry.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to settings YAML (defaults to configs/settings.yaml).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw input documents directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override processed output directory.",
    )
    parser.add_argument(
        "--add-file",
        type=Path,
        default=None,
        help="Copy a file into raw_dir before running ingestion (e.g., new program or synthetic test doc).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ingestion and print a concise summary to stdout."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        raw_dir = args.raw_dir.resolve() if args.raw_dir else cfg.paths.raw_dir
        out_dir = args.out_dir.resolve() if args.out_dir else cfg.paths.processed_dir

        if args.add_file is not None:
            docs, chunks, _, stats = add_file_and_reingest(
                args.add_file,
                raw_dir,
                out_dir,
                cfg.ingestion,
            )
        else:
            docs, chunks, _, stats = run_ingestion(raw_dir, out_dir, cfg.ingestion)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print("=== T-Education KB Ingestion Summary ===")
    print(f"Raw dir            : {raw_dir}")
    print(f"Output dir         : {out_dir}")
    print(f"Total raw files    : {stats.total_raw_files}")
    print(f"Unique active docs : {len(docs)}")
    print(f"Total chunks       : {len(chunks)}")
    print(f"Total clean chars  : {stats.total_clean_chars}")
    print(f"Avg/Max search len : {stats.avg_chunk_search_chars} / {stats.max_chunk_search_chars} chars")
    print(f"Duplicates skipped : {stats.duplicates_skipped}")
    print(f"Version conflicts  : {len(stats.version_conflicts)}")
    print(f"Ingestion errors   : {len(stats.errors)}")
    if stats.errors:
        for err in stats.errors:
            print(f"  - {err.source_file} [{err.error_type}]: {err.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
