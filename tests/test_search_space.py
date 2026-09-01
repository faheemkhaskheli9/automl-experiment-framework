"""Tests for the Phase 1 preprocessing search-space loader.

Each error-path test fails against a naive loader that trusts the YAML and
passes against ``src.search_space`` as written.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.search_space import (
    PreprocessingSearchSpace,
    SearchSpaceError,
    load_preprocessing_search_space,
)

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "preprocessing_search_space.yaml"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "space.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_shipped_config_loads_and_matches_readme_shape():
    space = load_preprocessing_search_space(SHIPPED_CONFIG)
    assert isinstance(space, PreprocessingSearchSpace)
    assert [s.name for s in space.slots] == [
        "imputation",
        "scaling",
        "categorical_encoding",
        "feature_selection",
    ]
    # imputation(2) x scaling(3+skip) x encoding(2) x selection(1+skip)
    assert space.combinations() == 2 * 4 * 2 * 2
    assert "Preprocessing search space" in space.describe()


def test_missing_file_is_hard_error(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="not found"):
        load_preprocessing_search_space(tmp_path / "nope.yaml")


def test_empty_file_is_error(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="empty"):
        load_preprocessing_search_space(_write(tmp_path, ""))


def test_invalid_yaml_is_error(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="invalid YAML"):
        load_preprocessing_search_space(_write(tmp_path, "slots: [unclosed\n"))


def test_non_mapping_top_level_is_error(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="mapping"):
        load_preprocessing_search_space(_write(tmp_path, "- just\n- a\n- list\n"))


def test_bare_mapping_without_preprocessing_key_is_accepted(tmp_path: Path):
    space = load_preprocessing_search_space(
        _write(
            tmp_path,
            """
            slots:
              - name: scaling
                candidates:
                  - name: standard
                    estimator: sklearn.preprocessing.StandardScaler
            """,
        )
    )
    assert space.combinations() == 1


def test_duplicate_slot_names_rejected(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="duplicate slot names"):
        load_preprocessing_search_space(
            _write(
                tmp_path,
                """
                preprocessing:
                  slots:
                    - name: scaling
                      candidates:
                        - {name: a, estimator: pkg.A}
                    - name: scaling
                      candidates:
                        - {name: b, estimator: pkg.B}
                """,
            )
        )


def test_duplicate_candidate_names_rejected(tmp_path: Path):
    with pytest.raises(SearchSpaceError, match="duplicate candidate names"):
        load_preprocessing_search_space(
            _write(
                tmp_path,
                """
                preprocessing:
                  slots:
                    - name: scaling
                      candidates:
                        - {name: dup, estimator: pkg.A}
                        - {name: dup, estimator: pkg.B}
                """,
            )
        )


def test_slot_needs_at_least_one_candidate(tmp_path: Path):
    with pytest.raises(SearchSpaceError):
        load_preprocessing_search_space(
            _write(
                tmp_path,
                """
                preprocessing:
                  slots:
                    - name: scaling
                      candidates: []
                """,
            )
        )


def test_unknown_key_is_rejected(tmp_path: Path):
    with pytest.raises(SearchSpaceError):
        load_preprocessing_search_space(
            _write(
                tmp_path,
                """
                preprocessing:
                  slots:
                    - name: scaling
                      typo_optional: true
                      candidates:
                        - {name: a, estimator: pkg.A}
                """,
            )
        )


@pytest.mark.parametrize(
    "param_block",
    [
        "{}",  # neither choices nor range
        "{choices: [1, 2], low: 0.0, high: 1.0}",  # both forms
        "{choices: []}",  # empty choices
        "{low: 1.0}",  # partial range
        "{low: 5.0, high: 1.0}",  # low > high
    ],
)
def test_bad_hyperparam_spec_rejected(tmp_path: Path, param_block: str):
    with pytest.raises(SearchSpaceError):
        load_preprocessing_search_space(
            _write(
                tmp_path,
                f"""
                preprocessing:
                  slots:
                    - name: scaling
                      candidates:
                        - name: standard
                          estimator: sklearn.preprocessing.StandardScaler
                          params:
                            with_mean: {param_block}
                """,
            )
        )
