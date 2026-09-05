from pathlib import Path

import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.model_search_space import (
    MODEL_REGISTRY,
    ModelCandidate,
    ModelSearchSpace,
    load_model_search_space,
)
from src.search_space import SearchSpaceError

CONFIG_PATH = Path("configs/model_search_space.yaml")


def test_example_config_loads_at_least_three_model_families():
    space = load_model_search_space(CONFIG_PATH)
    families = {m.model_family for m in space.models}
    assert len(families) >= 3
    assert families <= set(MODEL_REGISTRY)


def test_each_candidate_in_example_config_instantiates_a_real_estimator():
    space = load_model_search_space(CONFIG_PATH)
    instances = {m.name: m.instantiate() for m in space.models}
    assert isinstance(instances["rf_default"], RandomForestClassifier)
    assert isinstance(instances["logreg_default"], LogisticRegression)
    assert isinstance(instances["svm_default"], SVC)


def test_instantiated_estimator_actually_fits_and_predicts():
    from sklearn.datasets import make_classification

    space = load_model_search_space(CONFIG_PATH)
    candidate = next(m for m in space.models if m.name == "logreg_default")
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    model = candidate.instantiate()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_unknown_model_family_is_rejected():
    with pytest.raises(ValidationError):
        ModelCandidate(name="x", model_family="not_a_real_model")


def test_duplicate_candidate_names_are_rejected():
    with pytest.raises(ValidationError):
        ModelSearchSpace(
            models=[
                ModelCandidate(name="dupe", model_family="svm"),
                ModelCandidate(name="dupe", model_family="logistic_regression"),
            ]
        )


def test_missing_config_file_is_a_hard_error():
    with pytest.raises(SearchSpaceError):
        load_model_search_space("configs/does_not_exist.yaml")


def test_empty_config_file_is_a_hard_error(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(SearchSpaceError):
        load_model_search_space(empty)
