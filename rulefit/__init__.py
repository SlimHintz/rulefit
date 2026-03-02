"""RuleFit package public API."""

from ._conditions import RuleCondition
from .benchmark import (
    BenchmarkResult,
    benchmark_all,
    benchmark_rulefit_classification,
    benchmark_rulefit_regression,
    format_results_table,
)
from ._model import RuleFit, RuleFitClassifier, RuleFitRegressor
from ._rules import Rule, RuleEnsemble, extract_rules_from_tree
from ._scaling import FriedScale, Winsorizer

__all__ = [
    "RuleCondition",
    "Rule",
    "RuleEnsemble",
    "extract_rules_from_tree",
    "RuleFit",
    "RuleFitRegressor",
    "RuleFitClassifier",
    "FriedScale",
    "Winsorizer",
    "BenchmarkResult",
    "benchmark_rulefit_regression",
    "benchmark_rulefit_classification",
    "benchmark_all",
    "format_results_table",
]
