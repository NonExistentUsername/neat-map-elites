import time
from multiprocessing.pool import Pool
from typing import List, Optional

from mapelites.archive import GridArchive
from mapelites.interfaces import (
    IComparator,
    IFitnessFunction,
    IIndividualGenerator,
    IIndividualSelector,
)


class MapElites:
    def __init__(
        self,
        archive: GridArchive,
        fitness_funcion: IFitnessFunction,
        individual_selector: IIndividualSelector,
        individual_generator: IIndividualGenerator,
        fitness_comparator: IComparator,
        initial_size: int,
    ) -> None:
        self._archive = archive
        self._fitness_function = fitness_funcion
        self._individual_selector = individual_selector
        self._individual_generator = individual_generator
        self._fitness_comparator = fitness_comparator

        self.__fill_archive(initial_size)

    def __fill_archive(self, initial_size: int) -> None:
        with Pool(10) as p:
            individuals = self._individual_generator.create_individuals(initial_size)

            fitnesses = p.map(self._fitness_function, individuals)

            for individual, fitness in zip(individuals, fitnesses):
                self._archive.put_solution(individual, fitness)

    @property
    def solutions(self):
        return self._archive.solutions

    @property
    def best_solution(self):
        return self._archive.best_solution

    def run(
        self,
        iterations: int,
        batch_size: int,
        verbose: bool = True,
        stop_fitness: Optional[float] = None,
    ) -> List[float]:
        fitness_history: List[float] = []

        with Pool(10) as p:
            for iteration in range(iterations):
                iteration_start_time = time.time()

                individuals = self._individual_generator.get_mutated_individuals(
                    self._individual_selector, batch_size
                )
                fitnesses = p.map(self._fitness_function, individuals)

                for individual, fitness in zip(individuals, fitnesses):
                    if (
                        self._archive.put_solution(individual, fitness)
                        and stop_fitness
                        and self._fitness_comparator(stop_fitness, fitness.fitness)
                    ):
                        return fitness_history

                if verbose:
                    iteration_end_time = time.time()

                    print(f"Iteration: {iteration + 1}/{iterations}")
                    print(
                        f"Elapsed time: {iteration_end_time - iteration_start_time:.2f}s"
                    )
                    print(f"Archive fitness: {self._archive.fitness}")
                    print(f"Archive fullness: {self._archive.fullness*100.0:.2f}%")
                    print("")

                fitness_history.append(self._archive.fitness or 0.0)  # TODO

        return fitness_history
