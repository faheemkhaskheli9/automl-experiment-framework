"""Preprocessing search-space definitions loaded from YAML config.

Phase 1 scope: a *preprocessing search space* is an ordered list of **slots**
(e.g. imputation, scaling, encoding). Each slot offers one or more candidate
**transformers**; a later optimization loop (Phase 2) picks one candidate per
slot and samples its hyperparameters. This module only defines that structure
and loads it from a YAML file -- it does not build or fit any sklearn objects.

Design note (see ``docs/architecture.md``): pipelines are configuration-driven,
so experiments are reproducible from the YAML alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class SearchSpaceError(ValueError):
    """Raised when a search-space config cannot be read or parsed."""


class HyperParamSpec(BaseModel):
    """Candidate values for a single transformer hyperparameter.

    Provide **either** ``choices`` (categorical / discrete) **or** both ``low``
    and ``high`` (a numeric range), never both and never neither.
    """

    model_config = ConfigDict(extra="forbid")

    choices: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "HyperParamSpec":
        has_choices = self.choices is not None
        has_range = self.low is not None and self.high is not None
        partial_range = (self.low is None) != (self.high is None)
        if partial_range:
            raise ValueError("'low' and 'high' must both be set for a numeric range")
        if has_choices == has_range:
            raise ValueError(
                "specify either 'choices' or both 'low'/'high', not both or neither"
            )
        if has_choices and not self.choices:
            raise ValueError("'choices' must not be empty")
        if has_range and self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self


class TransformerCandidate(BaseModel):
    """One concrete transformer the optimizer may place in a slot."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    estimator: str = Field(
        ...,
        min_length=1,
        description="Import path, e.g. 'sklearn.preprocessing.StandardScaler'.",
    )
    params: dict[str, HyperParamSpec] = Field(default_factory=dict)


class PreprocessingSlot(BaseModel):
    """An ordered stage of the preprocessing pipeline with >=1 candidates."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    optional: bool = Field(
        default=False,
        description="If true, 'skip this slot entirely' is also a valid choice.",
    )
    candidates: list[TransformerCandidate] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_candidate_names(self) -> "PreprocessingSlot":
        names = [c.name for c in self.candidates]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"duplicate candidate names in slot '{self.name}': {dupes}"
            )
        return self


class PreprocessingSearchSpace(BaseModel):
    """Top-level preprocessing search space: an ordered list of slots."""

    model_config = ConfigDict(extra="forbid")

    slots: list[PreprocessingSlot] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_slot_names(self) -> "PreprocessingSearchSpace":
        names = [s.name for s in self.slots]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate slot names: {dupes}")
        return self

    def combinations(self) -> int:
        """Number of distinct transformer selections (ignoring hyperparameters)."""
        total = 1
        for slot in self.slots:
            total *= len(slot.candidates) + (1 if slot.optional else 0)
        return total

    def describe(self) -> str:
        """Human-readable multi-line summary for the CLI."""
        lines = [
            f"Preprocessing search space: {len(self.slots)} slot(s), "
            f"{self.combinations()} transformer combination(s)"
        ]
        for slot in self.slots:
            flag = " (optional)" if slot.optional else ""
            lines.append(f"  - {slot.name}{flag}: {len(slot.candidates)} candidate(s)")
            for cand in slot.candidates:
                pdesc = ", ".join(sorted(cand.params)) or "no tunable params"
                lines.append(f"      * {cand.name} [{cand.estimator}] -> {pdesc}")
        return "\n".join(lines)


def load_preprocessing_search_space(path: str | Path) -> PreprocessingSearchSpace:
    """Load and validate a preprocessing search space from a YAML file.

    The path is always caller-supplied, so a missing file is a hard error
    (robustness rule 7 -- never silently fall through to a default).

    Accepts either a top-level mapping with a ``preprocessing:`` key or a bare
    mapping that is itself the search space.
    """
    p = Path(path)
    if not p.is_file():
        raise SearchSpaceError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SearchSpaceError(f"invalid YAML in {p}: {exc}") from exc

    if raw is None:
        raise SearchSpaceError(f"config file is empty: {p}")
    if not isinstance(raw, dict):
        raise SearchSpaceError(
            f"expected a mapping at the top level of {p}, got {type(raw).__name__}"
        )

    section = raw.get("preprocessing", raw)
    try:
        return PreprocessingSearchSpace.model_validate(section)
    except ValidationError as exc:
        raise SearchSpaceError(
            f"invalid preprocessing search space in {p}:\n{exc}"
        ) from exc
