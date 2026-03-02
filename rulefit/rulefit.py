"""Backward-compatible import surface for legacy users.

Historically all implementations lived in this module. The package now uses
smaller internal modules and re-exports the same public symbols here.
"""

from ._conditions import RuleCondition
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
]
