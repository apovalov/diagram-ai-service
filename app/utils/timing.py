from __future__ import annotations

from time import perf_counter

__all__ = ["Timer"]


class Timer:
    """Context manager to measure elapsed seconds.

    Usage:
        with Timer() as t:
            ...
        t.elapsed_s
    """

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        self.elapsed_s: float = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_s = perf_counter() - self._start
