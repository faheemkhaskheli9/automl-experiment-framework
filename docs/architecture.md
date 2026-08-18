# Architecture Notes: AutoML Experimentation Pipeline

## Pipeline

```text
Dataset -> Preprocessing Search Space -> Model Search Space -> Hyperparameter Optimization (Optuna) -> Tracked Experiments -> Leaderboard
```

## Components

- Automated model selection
- Preprocessing pipeline search
- Feature engineering options
- Hyperparameter tuning
- Cross-validation
- Experiment tracking
- Leaderboard of results

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
