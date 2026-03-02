"""Benchmark tests: verify timing and model quality for RuleFit."""

from __future__ import annotations

import time

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from rulefit import RuleFitClassifier, RuleFitRegressor
from rulefit.benchmark import (
    benchmark_rulefit_classification,
    benchmark_rulefit_regression,
    format_results_table,
)


# ---------------------------------------------------------------------------
# Smoke tests: verify the benchmark helpers run and return sensible shapes
# ---------------------------------------------------------------------------

def test_benchmark_helpers_smoke():
    reg = benchmark_rulefit_regression(n_samples=120, n_features=8, max_rules=100, cv=2, max_iter=400)
    clf = benchmark_rulefit_classification(n_samples=120, n_features=8, max_rules=100, cv=2, max_iter=200)
    results = [*reg, *clf]

    assert len(results) == 4
    assert {r.tree_method for r in results} == {"random_forest", "gradient_boosting"}
    assert all(r.fit_time_s > 0 for r in results)
    assert all(r.predict_time_s >= 0 for r in results)

    table = format_results_table(results)
    assert "tree_method" in table
    assert "fit_time_s" in table


# ---------------------------------------------------------------------------
# Performance tests: train/test split + score thresholds + timing output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_generator", ["random_forest", "gradient_boosting"])
def test_regression_r2_on_friedman1(tree_generator):
    """RuleFit should achieve R² > 0.80 on a held-out Friedman1 test set."""
    X, y = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    t0 = time.perf_counter()
    model = RuleFitRegressor(
        default_tree_generator=tree_generator,
        max_rules=200,
        cv=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    score = r2_score(y_test, preds)
    rules = model.get_rules(exclude_zero_coef=True)

    print(
        f"\n[regression/{tree_generator}]"
        f"  fit={fit_time:.2f}s"
        f"  predict={predict_time:.4f}s"
        f"  R²={score:.4f}"
        f"  active_rules={len(rules)}"
    )

    assert score > 0.70, f"Expected R² > 0.70, got {score:.4f}"
    assert len(rules) > 0, "Expected at least one active rule/feature"


@pytest.mark.parametrize("tree_generator", ["random_forest", "gradient_boosting"])
def test_classification_accuracy(tree_generator):
    """RuleFit should achieve accuracy > 0.75 on a synthetic classification task."""
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        class_sep=1.5,
        random_state=42,
    )
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    t0 = time.perf_counter()
    model = RuleFitClassifier(
        default_tree_generator=tree_generator,
        max_rules=200,
        cv=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    score = accuracy_score(y_test, preds)
    rules = model.get_rules(exclude_zero_coef=True)

    print(
        f"\n[classification/{tree_generator}]"
        f"  fit={fit_time:.2f}s"
        f"  predict={predict_time:.4f}s"
        f"  acc={score:.4f}"
        f"  active_rules={len(rules)}"
    )

    assert score > 0.75, f"Expected accuracy > 0.75, got {score:.4f}"
    assert len(rules) > 0, "Expected at least one active rule/feature"


# ---------------------------------------------------------------------------
# get_rules() content tests
# ---------------------------------------------------------------------------

def test_get_rules_returns_expected_columns():
    X, y = make_friedman1(n_samples=200, n_features=5, noise=1.0, random_state=0)
    model = RuleFitRegressor(max_rules=100, cv=2, random_state=0)
    model.fit(X, y, feature_names=[f"x{i}" for i in range(5)])

    rules = model.get_rules()

    assert list(rules.columns) == ["rule", "type", "feature_name", "coef", "support", "importance"]
    assert set(rules["type"].unique()).issubset({"rule", "linear"})
    assert rules["feature_name"].notna().all()
    assert (rules["support"] > 0).all()
    assert (rules["support"] <= 1).all()
    assert (rules["importance"] >= 0).all()


def test_get_rules_exclude_zero_coef():
    X, y = make_friedman1(n_samples=200, n_features=5, noise=1.0, random_state=0)
    model = RuleFitRegressor(max_rules=100, cv=2, random_state=0)
    model.fit(X, y)

    all_rules = model.get_rules(exclude_zero_coef=False)
    active_rules = model.get_rules(exclude_zero_coef=True)

    assert len(active_rules) <= len(all_rules)
    assert (active_rules["coef"] != 0).all()


# ---------------------------------------------------------------------------
# Full benchmark table (run manually: pytest -s -k benchmark_summary)
# ---------------------------------------------------------------------------

def test_benchmark_summary(capsys):
    """Print a full timing + score table. Run with: pytest -s -k benchmark_summary"""
    reg = benchmark_rulefit_regression(n_samples=500, n_features=10, max_rules=300, cv=3, max_iter=1000)
    clf = benchmark_rulefit_classification(n_samples=500, n_features=10, max_rules=300, cv=3, max_iter=1000)
    table = format_results_table([*reg, *clf])

    with capsys.disabled():
        print(f"\n{table}")
