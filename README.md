# RuleFit

Implementation of the rule-based prediction algorithm from
[Friedman & Popescu (2005)](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf),
with a scikit-learn compatible API supporting both regression and classification.

## How it works

1. **Fit a tree ensemble** (Random Forest or Gradient Boosting) on the training data
2. **Extract binary rules** from every path in every tree — each rule is a conjunction of feature conditions (e.g. `age > 35 & income <= 80000`)
3. **Fit an L1-regularized linear model** (Lasso for regression, logistic regression for classification) on the rule activations combined with the original features

The L1 penalty drives most rule coefficients to zero, leaving a sparse, interpretable model.

## Requirements

- Python ≥ 3.9
- scikit-learn ≥ 1.8
- numpy ≥ 1.22
- pandas ≥ 1.5

## Installation

```bash
pip install git+https://github.com/SlimHintz/rulefit.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/SlimHintz/rulefit.git
cd rulefit
pip install -e .
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate rulefit
```

## Usage

### Regression

```python
import pandas as pd
from rulefit import RuleFitRegressor

data = pd.read_csv("boston.csv", index_col=0)
y = data.medv.values
X = data.drop("medv", axis=1)

rf = RuleFitRegressor(random_state=42)
rf.fit(X.to_numpy(), y, feature_names=list(X.columns))

predictions = rf.predict(X.to_numpy())
```

### Classification

```python
from rulefit import RuleFitClassifier

clf = RuleFitClassifier(random_state=42)
clf.fit(X_train, y_train, feature_names=feature_names)

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### Choosing a tree generator

Both classes default to Random Forest internally. Switch to Gradient Boosting
or supply your own fitted ensemble:

```python
# Built-in gradient boosting
rf = RuleFitRegressor(default_tree_generator="gradient_boosting")

# Custom sklearn estimator
from sklearn.ensemble import GradientBoostingRegressor
rf = RuleFitRegressor(
    tree_generator=GradientBoostingRegressor(n_estimators=500, learning_rate=0.01)
)
```

### Model type

Control whether the linear model uses rules, raw linear features, or both:

```python
rf = RuleFitRegressor(model_type="rl")  # rules + linear (default)
rf = RuleFitRegressor(model_type="r")   # rules only
rf = RuleFitRegressor(model_type="l")   # linear only
```

## Inspecting rules

`get_rules()` returns a DataFrame describing every rule and linear term in the
fitted model:

```python
rules = rf.get_rules()
print(rules.columns)
# ['rule', 'type', 'feature_name', 'coef', 'support', 'importance']
```

| Column | Description |
|---|---|
| `rule` | Human-readable condition string, e.g. `age > 35 & income <= 80000` |
| `type` | `"rule"` or `"linear"` |
| `feature_name` | Sorted, underscore-joined feature names involved (e.g. `age_income`); collision-enumerated when multiple rules share the same features (e.g. `age_income_0`, `age_income_1`) |
| `coef` | Coefficient from the linear model (sign = direction, magnitude = effect size) |
| `support` | Fraction of training samples for which the rule is active (`1.0` for linear terms) |
| `importance` | `abs(coef) × support` — a principled ranking metric |

### Common patterns

```python
# Active rules only, sorted by importance
active = rf.get_rules(exclude_zero_coef=True).sort_values("importance", ascending=False)

# Control threshold rounding (default: 2 decimal places)
rules = rf.get_rules(round_digits=4)   # more precision
rules = rf.get_rules(round_digits=None)  # raw float splits

# Group rules by which features they involve
by_feature = active.groupby("feature_name")["importance"].sum().sort_values(ascending=False)
```

Example output:

```
rule                                    type    feature_name       coef   support  importance
age                                     linear  age               1.36    1.00     1.36
income                                  linear  income            0.82    1.00     0.82
age > 0.35 & income <= 0.48             rule    age_income_0      1.21    0.43     0.52
age > 0.41 & income <= 0.52 & cr... 0  rule    age_credit_income  0.94   0.31     0.29
```

## Benchmarking

Compare Random Forest vs Gradient Boosting tree generation:

```bash
python benchmark_tree_generators.py --reg-samples 3000 --clf-samples 3000
```

Or run the benchmark tests directly (prints a timing + score table):

```bash
pytest -s -k benchmark_summary
```

## Running tests

```bash
pytest
```

## Changelog

### [v0.3] - 2025

- Modularized package into `_model`, `_rules`, `_conditions`, `_scaling`, `_ordered_set` modules
- Added `get_rules()` returning a DataFrame with `rule`, `type`, `feature_name`, `coef`, `support`, `importance`
- `feature_name` column: sorted underscore-joined feature names, collision-enumerated for rules
- `round_digits` parameter on `get_rules()` (default: 2) for readable threshold display
- `exclude_zero_coef` parameter on `get_rules()` (default: `False`)
- Added `RuleFitRegressor` and `RuleFitClassifier` convenience subclasses
- Aligned `max_iter` default to 1000 for both regressor and classifier
- Fixed sklearn 1.8 deprecations: `LassoCV` `n_alphas`, `LogisticRegressionCV` `penalty`
- Added `environment.yml` for conda environment setup
- Added `pyproject.toml` with build config and pytest settings
- Added performance and timing benchmark tests

### [v0.2] - 2017-11-24

- Introduced classification support
- Added Friedscale variable scaling
- Added random-size tree generation (`exp_rand_tree_size`)

### [v0.1] - 2016-06-18

- Initial release
