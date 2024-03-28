import abc
from functools import reduce
from typing import Any, List, Optional, Tuple

from mapelites.interfaces import (
    Fitness,
    IComparator,
    IFitnessFunction,
    IIndividualGenerator,
)


class Bin:
    def __init__(self):
        self.solution = None
        self.fitness: float = float("-inf")  # type: ignore


class GridArchive:
    def __init__(
        self,
        dimensions: List[int],
        ranges: List[Tuple[float, float]],
        comparator: IComparator,
    ):
        if len(dimensions) != len(ranges):
            raise ValueError("Dimensions and ranges must have the same length")

        self._dimensions = dimensions
        self._cells_count = reduce(lambda x, y: x * y + x, dimensions) + 1
        self._bins: List[Bin] = [Bin() for _ in range(self._cells_count)]
        self._ranges = ranges
        self._comparator = comparator

        self._fitness: Optional[float] = None
        self._best_solution: Optional[Any] = None

    def _get_bin_index(self, fitness_obj: Fitness) -> int:
        coords = [
            int(
                (fitness_obj.features_fitness[i] - self._ranges[i][0])
                / (self._ranges[i][1] - self._ranges[i][0])
                * self._dimensions[i]
            )  # scale the feature to the range and then to the grid
            for i in range(len(self._dimensions))
        ]
        bin_index = 0
        k = 1
        for i in range(len(self._dimensions)):
            bin_index += coords[i] * k
            k *= self._dimensions[i]
        return bin_index

    @property
    def cells(self) -> List[Bin]:
        return self._bins

    @property
    def non_empty_cells(self) -> List[Bin]:
        return [bin for bin in self._bins if bin.solution is not None]

    @property
    def solutions(self) -> List[Any]:
        return [bin.solution for bin in self._bins if bin.solution is not None]

    @property
    def best_solution(self) -> Optional[Any]:
        return self._best_solution

    def get_solution(self, cell_idx: int) -> Optional[List[float]]:
        return self._bins[cell_idx].solution

    @property
    def fitness(self) -> Optional[float]:
        return self._fitness

    @property
    def fullness(self) -> float:
        return sum(bin.solution is not None for bin in self._bins) / self._cells_count

    def put_solution(self, solution: List[float], fitness_obj: Fitness) -> bool:
        bin_index = self._get_bin_index(fitness_obj)

        if bin_index < 0 or bin_index >= self._cells_count:
            print(f"bin_index: {bin_index}")
            print(f"fitness_obj.features_fitness: {fitness_obj.features_fitness}")
            print(f"ranges: {self._ranges}")
            print(f"dimensions: {self._dimensions}")
            raise ValueError("Bin index out of range")

        if self._bins[bin_index].solution is None or self._comparator(
            self._bins[bin_index].fitness, fitness_obj.fitness
        ):
            self._bins[bin_index].solution = solution
            self._bins[bin_index].fitness = fitness_obj.fitness
            if self._fitness is None or self._comparator(
                self._fitness, fitness_obj.fitness
            ):
                self._fitness = fitness_obj.fitness
                self._best_solution = solution
            return True

        return False
