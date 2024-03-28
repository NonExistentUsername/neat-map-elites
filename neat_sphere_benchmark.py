from typing import Any

import numpy as np

from mapelites import *
from mapelites.interfaces import Fitness, IIndividualSelector

IndividualType = np.ndarray


class SphereFitness(IFitnessFunction):
    def __call__(self, individual: Any) -> Fitness:
        if not isinstance(individual, IndividualType):
            raise ValueError("Individual must be a numpy array")

        x: float = individual[0]
        y: float = individual[1]

        return Fitness(
            features_fitness=[max(min(x, 5.12), -5.12), max(min(y, 5.12), -5.12)],
            fitness=x**2 + y**2,
        )


class IndividualGenerator(IIndividualGenerator):
    def create_individuals(self, count: int) -> List[IndividualType]:
        return [np.random.rand(2) for _ in range(count)]

    def get_mutated_individuals(
        self, individual_selector: IIndividualSelector, count: int
    ) -> Any:
        return [
            individual_selector.get() + np.random.normal(0, 0.1, size=2)
            for _ in range(count)
        ]


def mut_gaussian(
    individual: Any, mu: float = 0.0, sigma: float = 1.0, mut_pb: float = 0.2
):
    for i in range(len(individual)):
        if random.random() < mut_pb:
            individual[i] += random.gauss(mu, sigma)
    return individual


class IndividualGaussianGenerator(IIndividualGenerator):
    def create_individuals(self, count: int) -> List[IndividualType]:
        return [np.random.rand(2) for _ in range(count)]

    def get_mutated_individuals(
        self, individual_selector: IIndividualSelector, count: int
    ) -> Any:
        return [mut_gaussian(individual_selector.get()) for _ in range(count)]


def create(generator_type: str):

    arch = GridArchive(
        dimensions=[32, 32],
        ranges=[(-5.12, 5.12), (-5.12, 5.12)],
        comparator=MinComparator(),
    )

    generator = None
    if generator_type == "gaussian":
        generator = IndividualGaussianGenerator()
    else:
        generator = IndividualGenerator()

    map_elites = MapElites(
        archive=arch,
        fitness_funcion=SphereFitness(),
        individual_selector=RandomIndividualSelector(arch),
        individual_generator=generator,
        initial_size=32,
    )

    return arch, map_elites


tests = 32
summaries = [0, 0]

for test_index in range(tests):
    print(f"Test {test_index + 1}/{tests}")

    arch, map_elites = create("default")

    map_elites.run(iterations=1024, batch_size=32, verbose=False)

    summaries[0] += arch.fullness

for test_index in range(tests):
    print(f"Test {test_index + 1}/{tests}")

    arch, map_elites = create("gaussian")

    map_elites.run(iterations=1024, batch_size=32, verbose=False)

    summaries[1] += arch.fullness

print(f"Average fullness (default): {summaries[0] / tests}")
print(f"Average fullness (gaussian): {summaries[1] / tests}")
