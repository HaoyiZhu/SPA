from collections.abc import Iterable
from typing import List, Optional, Union

from lightning.fabric.utilities.data import sized_len
from lightning.pytorch.utilities.combined_loader import (
    _ITERATOR_RETURN,
    _SUPPORTED_MODES,
    CombinedLoader,
    _CombinationMode,
    _ModeIterator,
)
from typing_extensions import Self, override


class _CyclicSequential(_ModeIterator):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        super().__init__(iterables, limits)
        self._consumed: List[int] = []
        self._length: List[int] = []
        self._iterator_idx = 0

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        find = False
        for i in range(n):
            if self._consumed[self._iterator_idx] < self._length[self._iterator_idx]:
                out[self._iterator_idx] = next(self.iterators[self._iterator_idx])
                self._consumed[self._iterator_idx] += 1
                find = True
                break
            self._iterator_idx = (self._iterator_idx + 1) % n
        assert find, "All iterators are exhausted."

        iterator_idx = self._iterator_idx
        self._iterator_idx = (self._iterator_idx + 1) % n
        batch_idx = self._idx
        self._idx += 1
        return out, batch_idx, iterator_idx

    @override
    def __iter__(self) -> Self:
        super().__iter__()
        self._consumed = [0] * len(self.iterables)
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            lengths = [
                min(length, limit) for length, limit in zip(lengths, self.limits)
            ]
        self._length = lengths
        self._iterator_idx = 0
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return sum(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[misc]
        return sum(lengths)  # type: ignore[arg-type]

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []
        self._length = []
        self._iterator_idx = 0


_SUPPORTED_MODES["max_size_cycle"] = _CombinationMode(
    fn=sum, iterator=_CyclicSequential
)


def _get_iterables_lengths(iterables: List[Iterable]) -> List[Union[int, float]]:
    return [
        (float("inf") if (length := sized_len(iterable)) is None else length)
        for iterable in iterables
    ]
