"""Benchmark helpers for comparing RuleFit tree-generator strategies."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import accuracy_score, r2_score

from ._model import RuleFit


@dataclass
class BenchmarkResult:
    task: str
    tree_method: str
    fit_time_s: float
    predict_time_s: float
    score_name: str
    score_value: float
    n_samples: int
    n_features: int

    def to_dict(self) -> dict:
        return asdict(self)


def _time_model(run: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = perf_counter()
    out = run()
    elapsed = perf_counter() - start
    return out, elapsed


def benchmark_rulefit_regression(
    *,
    n_samples: int = 5000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 1.0,
    random_state: int = 42,
    max_rules: int = 1000,
    tree_size: int = 5,
    cv: int = 3,
    max_iter: int = 2000,
) -> list[BenchmarkResult]:
    """Compare random-forest vs gradient-boosting tree generation for regression."""
    x, y = make_friedman1(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    # Keep only a subset informative by appending noise dimensions when needed.
    if n_informative < n_features:
        rng = np.random.RandomState(random_state)
        x[:, n_informative:] = rng.normal(size=(n_samples, n_features - n_informative))

    results = []
    for method in ("random_forest", "gradient_boosting"):
        model = RuleFit(
            rfmode="regress",
            default_tree_generator=method,
            max_rules=max_rules,
            tree_size=tree_size,
            cv=cv,
            max_iter=max_iter,
            random_state=random_state,
            exp_rand_tree_size=False,
            model_type="rl",
            n_jobs=1,
        )

        _, fit_time = _time_model(lambda: model.fit(x, y))
        pred, pred_time = _time_model(lambda: model.predict(x))

        results.append(
            BenchmarkResult(
                task="regression",
                tree_method=method,
                fit_time_s=fit_time,
                predict_time_s=pred_time,
                score_name="r2",
                score_value=r2_score(y, pred),
                n_samples=n_samples,
                n_features=n_features,
            )
        )
    return results


def benchmark_rulefit_classification(
    *,
    n_samples: int = 5000,
    n_features: int = 25,
    random_state: int = 42,
    max_rules: int = 1000,
    tree_size: int = 5,
    cv: int = 3,
    max_iter: int = 500,
) -> list[BenchmarkResult]:
    """Compare random-forest vs gradient-boosting tree generation for classification."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 6),
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        random_state=random_state,
    )
    y = np.where(y == 1, 1, -1)

    results = []
    for method in ("random_forest", "gradient_boosting"):
        model = RuleFit(
            rfmode="classify",
            default_tree_generator=method,
            max_rules=max_rules,
            tree_size=tree_size,
            cv=cv,
            max_iter=max_iter,
            random_state=random_state,
            exp_rand_tree_size=False,
            model_type="rl",
            n_jobs=1,
        )

        _, fit_time = _time_model(lambda: model.fit(x, y))
        pred, pred_time = _time_model(lambda: model.predict(x))

        results.append(
            BenchmarkResult(
                task="classification",
                tree_method=method,
                fit_time_s=fit_time,
                predict_time_s=pred_time,
                score_name="accuracy",
                score_value=accuracy_score(y, pred),
                n_samples=n_samples,
                n_features=n_features,
            )
        )
    return results


def benchmark_all(**kwargs) -> list[BenchmarkResult]:
    """Run both regression and classification benchmarks."""
    regression_kwargs = {
        key[len("reg_") :]: value
        for key, value in kwargs.items()
        if key.startswith("reg_")
    }
    classification_kwargs = {
        key[len("clf_") :]: value
        for key, value in kwargs.items()
        if key.startswith("clf_")
    }

    return [
        *benchmark_rulefit_regression(**regression_kwargs),
        *benchmark_rulefit_classification(**classification_kwargs),
    ]


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a plain-text table."""
    headers = [
        "task",
        "tree_method",
        "fit_time_s",
        "predict_time_s",
        "score_name",
        "score_value",
        "n_samples",
        "n_features",
    ]
    rows = [headers]
    for r in results:
        rows.append(
            [
                r.task,
                r.tree_method,
                f"{r.fit_time_s:.4f}",
                f"{r.predict_time_s:.4f}",
                r.score_name,
                f"{r.score_value:.4f}",
                str(r.n_samples),
                str(r.n_features),
            ]
        )

    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
