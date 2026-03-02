import numpy as np
import pytest
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor

from rulefit import RuleFit


def test_rulefit_regression_recovers_rule_structure():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(1200, 6))

    y = (
        2.0 * (x[:, 0] > 0.25)
        - 1.6 * (x[:, 1] < -0.4)
        + 0.8 * x[:, 2]
        + rng.normal(scale=0.05, size=x.shape[0])
    )

    model = RuleFit(
        rfmode="regress",
        random_state=42,
        max_rules=400,
        tree_size=5,
        cv=3,
        max_iter=2000,
        model_type="rl",
    )
    model.fit(x, y)
    pred = model.predict(x)

    assert r2_score(y, pred) > 0.93


def test_rulefit_classification_predicts_nonlinear_threshold_target():
    rng = np.random.RandomState(7)
    x = rng.uniform(-1.0, 1.0, size=(1500, 5))
    boundary = ((x[:, 0] > 0.3) & (x[:, 1] < -0.2)) | (x[:, 2] > 0.65)
    y = np.where(boundary, 1, -1)

    model = RuleFit(
        rfmode="classify",
        random_state=7,
        max_rules=400,
        tree_size=5,
        cv=3,
        max_iter=500,
        model_type="rl",
    )
    model.fit(x, y)

    pred = model.predict(x)
    proba = model.predict_proba(x)

    assert accuracy_score(y, pred) > 0.95
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_default_tree_generator_strategy_is_configurable():
    model = RuleFit(rfmode="regress", default_tree_generator="gradient_boosting")
    model._init_default_tree_generator(n_samples=100)
    assert model.tree_generator.__class__.__name__ == "GradientBoostingRegressor"


def test_explicit_tree_generator_override_is_preserved():
    custom = RandomForestRegressor(n_estimators=5, random_state=0)
    model = RuleFit(rfmode="regress", tree_generator=custom)

    x = np.random.RandomState(0).normal(size=(80, 3))
    y = x[:, 0] - x[:, 1]
    model.fit(x, y)

    assert model.tree_generator is custom


def test_invalid_default_tree_generator_raises():
    model = RuleFit(default_tree_generator="bad-generator")
    with pytest.raises(ValueError, match="default_tree_generator"):
        model._init_default_tree_generator(n_samples=100)
