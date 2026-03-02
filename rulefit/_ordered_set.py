"""Minimal ordered set implementation used internally.

This avoids requiring the external ``ordered-set`` dependency while
preserving the small API surface used by this package.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class OrderedSet(Generic[T]):
    """Set that preserves insertion order."""

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._data: Dict[T, None] = {}
        if iterable is not None:
            self.update(iterable)

    def add(self, item: T) -> None:
        self._data[item] = None

    def update(self, iterable: Iterable[T]) -> None:
        for item in iterable:
            self._data[item] = None

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __repr__(self) -> str:
        values = ", ".join(repr(item) for item in self._data)
        return f"OrderedSet([{values}])"
