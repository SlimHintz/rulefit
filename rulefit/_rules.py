"""Rule extraction and transformation primitives."""

from __future__ import annotations

from functools import reduce
from typing import Callable

import numpy as np

from ._conditions import RuleCondition
from ._ordered_set import OrderedSet


class Rule:
    """Binary rule composed of one or more conditions."""

    def __init__(self, rule_conditions: list[RuleCondition], prediction_value: float):
        self.conditions = OrderedSet(rule_conditions)
        self.support = min(c.support for c in rule_conditions)
        self.prediction_value = prediction_value
        self.rule_direction = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply all conditions to samples and return a binary vector."""
        applies = [condition.transform(x) for condition in self.conditions]
        return reduce(lambda left, right: left * right, applies)

    def __str__(self) -> str:
        return " & ".join(str(condition) for condition in self.conditions)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return sum(hash(condition) for condition in self.conditions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return hash(self) == hash(other)


def extract_rules_from_tree(tree, feature_names=None) -> OrderedSet[Rule]:
    """Turn a fitted sklearn tree into a set of rules."""
    rules: OrderedSet[Rule] = OrderedSet()

    def traverse_nodes(node_id=0, operator=None, threshold=None, feature=None, conditions=None):
        conditions = [] if conditions is None else conditions

        if node_id != 0:
            feature_name = feature_names[feature] if feature_names is not None else feature
            rule_condition = RuleCondition(
                feature_index=feature,
                threshold=threshold,
                operator=operator,
                support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                feature_name=feature_name,
            )
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []

        if tree.children_left[node_id] != tree.children_right[node_id]:
            split_feature = tree.feature[node_id]
            split_threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", split_threshold, split_feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", split_threshold, split_feature, new_conditions)
            return

        if new_conditions:
            rules.add(Rule(new_conditions, tree.value[node_id][0][0]))

    traverse_nodes()
    return rules


class RuleEnsemble:
    """Ensemble of binary rules extracted from decision trees."""

    def __init__(self, tree_list, feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = OrderedSet()
        self._extract_rules()
        self.rules = list(self.rules)

    def _extract_rules(self) -> None:
        for tree in self.tree_list:
            tree_obj = tree[0] if isinstance(tree, (list, tuple, np.ndarray)) else tree
            rules = extract_rules_from_tree(tree_obj.tree_, feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func: Callable[[Rule], bool]) -> None:
        self.rules = [rule for rule in self.rules if func(rule)]

    def filter_short_rules(self, k: int) -> None:
        self.filter_rules(lambda rule: len(rule.conditions) > k)

    def transform(self, x: np.ndarray, coefs: np.ndarray | None = None) -> np.ndarray:
        """Transform input into a matrix where each column is one rule activation."""
        rule_list = list(self.rules)
        n_rules = len(rule_list)
        if n_rules == 0:
            return np.zeros((x.shape[0], 0), dtype=np.int8)

        if coefs is None:
            return np.column_stack([rule.transform(x) for rule in rule_list])

        coef_arr = np.asarray(coefs)
        active_mask = coef_arr != 0
        result = np.zeros((x.shape[0], n_rules), dtype=np.int8)
        if np.any(active_mask):
            active_idx = np.flatnonzero(active_mask)
            result[:, active_mask] = np.column_stack([
                rule_list[i].transform(x) for i in active_idx
            ])
        return result

    def __str__(self) -> str:
        return str([str(rule) for rule in self.rules])
