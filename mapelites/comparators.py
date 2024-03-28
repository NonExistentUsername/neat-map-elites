from mapelites.interfaces import IComparator


class MaxComparator(IComparator):
    def __call__(self, a: float, b: float) -> bool:
        return a < b


class MinComparator(IComparator):
    def __call__(self, a: float, b: float) -> bool:
        return a > b
