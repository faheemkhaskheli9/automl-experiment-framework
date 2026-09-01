"""Command-line entrypoint for the AutoML experimentation framework.

Phase 1 exposes a single command: load a preprocessing search-space YAML config
and print a summary. Later phases add optimization and leaderboard commands.

    python -m src.main show-preprocessing --config configs/preprocessing_search_space.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.search_space import SearchSpaceError, load_preprocessing_search_space

DEFAULT_CONFIG = Path("configs/preprocessing_search_space.yaml")


def _cmd_show_preprocessing(args: argparse.Namespace) -> int:
    try:
        space = load_preprocessing_search_space(args.config)
    except SearchSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(space.describe())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser(
        "show-preprocessing",
        help="Load a preprocessing search space from YAML and print a summary.",
    )
    p_show.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the preprocessing search-space YAML (default: {DEFAULT_CONFIG}).",
    )
    p_show.set_defaults(func=_cmd_show_preprocessing)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
