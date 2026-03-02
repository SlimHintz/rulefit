"""Rule condition primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuleCondition:
    """Binary rule condition used as a building block for rules."""

    feature_index: int
    threshold: float
    operator: str
    support: float
    feature_name: str | None = None

    def __str__(self) -> str:
        feature = self.feature_name if self.feature_name is not None else self.feature_index
        return f"{feature} {self.operator} {self.threshold}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuleCondition):
            return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        # Keep legacy semantics: support does not determine rule identity.
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the condition to all rows in ``x`` and return {0, 1}."""
        column = x[:, self.feature_index]
        if self.operator == "<=":
            return (column <= self.threshold).astype(np.int8, copy=False)
        if self.operator == ">":
            return (column > self.threshold).astype(np.int8, copy=False)
        raise ValueError(f"Unsupported operator: {self.operator}")
