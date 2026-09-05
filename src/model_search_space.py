"""Model search-space definitions loaded from YAML config.

A *model search space* is a list of **model candidates** — one per model
family (RandomForest, GradientBoosting, LogisticRegression, SVM, ...) — each
with its own named hyperparameters and sample ranges/choices, reusing
:class:`~src.search_space.HyperParamSpec`. A ``MODEL_REGISTRY`` maps each
config-level model-family name to the actual scikit-learn estimator class, so
growing the space is a registry entry (data), not a new Python class per model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.search_space import HyperParamSpec, SearchSpaceError

# Registry: config-level model-family name -> scikit-learn estimator class.
# Add a model family by adding an entry here, not a new Python type.
MODEL_REGISTRY: dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
}


class ModelCandidate(BaseModel):
    """One model family the optimizer may select, with its hyperparameter space."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    model_family: str = Field(
        ..., description=f"one of: {', '.join(sorted(MODEL_REGISTRY))}"
    )
    params: dict[str, HyperParamSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _model_family_is_registered(self) -> "ModelCandidate":
        if self.model_family not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown model_family '{self.model_family}'; registered families: "
                f"{sorted(MODEL_REGISTRY)}"
            )
        return self

    def estimator_class(self) -> type:
        return MODEL_REGISTRY[self.model_family]

    def default_params(self) -> dict[str, Any]:
        """One representative value per hyperparameter (first choice, or ``low``).

        Not a tuned configuration -- just enough to prove the candidate wires up
        to a real, instantiable estimator. Optuna (Phase 2) does the actual sampling.
        """
        values: dict[str, Any] = {}
        for name, spec in self.params.items():
            values[name] = spec.choices[0] if spec.choices is not None else spec.low
        return values

    def instantiate(self):
        """Build a real scikit-learn estimator instance from this candidate."""
        return self.estimator_class()(**self.default_params())


class ModelSearchSpace(BaseModel):
    """Top-level model search space: a list of candidate model families."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelCandidate] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "ModelSearchSpace":
        names = [m.name for m in self.models]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate model candidate names: {dupes}")
        return self

    def describe(self) -> str:
        lines = [f"Model search space: {len(self.models)} model candidate(s)"]
        for model in self.models:
            pdesc = ", ".join(sorted(model.params)) or "no tunable params"
            lines.append(f"  - {model.name} [{model.model_family}] -> {pdesc}")
        return "\n".join(lines)


def load_model_search_space(path: str | Path) -> ModelSearchSpace:
    """Load and validate a model search space from a YAML file.

    The path is always caller-supplied, so a missing file is a hard error
    (robustness rule 7 -- never silently fall through to a default).

    Accepts either a top-level mapping with a ``models:`` key or a bare mapping
    that is itself ``{"models": [...]}``.
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

    # Support both a dedicated file (top level IS {"models": [...]}) and a
    # combined config that also carries a "preprocessing:" section alongside.
    section = {"models": raw["models"]} if "models" in raw else raw
    try:
        return ModelSearchSpace.model_validate(section)
    except ValidationError as exc:
        raise SearchSpaceError(f"invalid model search space in {p}:\n{exc}") from exc
