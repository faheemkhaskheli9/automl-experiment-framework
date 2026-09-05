"""Command-line entrypoint for the AutoML experimentation framework.

Phase 1 exposes commands to load a search-space YAML config and print a
summary. Later phases add optimization and leaderboard commands.

    python -m src.main show-preprocessing --config configs/preprocessing_search_space.yaml
    python -m src.main show-models --config configs/model_search_space.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.model_search_space import load_model_search_space
from src.search_space import SearchSpaceError, load_preprocessing_search_space

DEFAULT_PREPROCESSING_CONFIG = Path("configs/preprocessing_search_space.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/model_search_space.yaml")


def _cmd_show_preprocessing(args: argparse.Namespace) -> int:
    try:
        space = load_preprocessing_search_space(args.config)
    except SearchSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(space.describe())
    return 0


def _cmd_show_models(args: argparse.Namespace) -> int:
    try:
        space = load_model_search_space(args.config)
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
        default=DEFAULT_PREPROCESSING_CONFIG,
        help=f"Path to the preprocessing search-space YAML (default: {DEFAULT_PREPROCESSING_CONFIG}).",
    )
    p_show.set_defaults(func=_cmd_show_preprocessing)

    p_models = sub.add_parser(
        "show-models",
        help="Load a model search space from YAML and print a summary.",
    )
    p_models.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Path to the model search-space YAML (default: {DEFAULT_MODEL_CONFIG}).",
    )
    p_models.set_defaults(func=_cmd_show_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
