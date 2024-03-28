import uuid
from typing import Any

import numpy as np
from neat import reproduction  # type: ignore
from neat import config, genome, nn, population, species, stagnation

from mapelites import *
from mapelites.interfaces import Fitness, IIndividualSelector

IndividualType = genome.DefaultGenome


neat_config = config.Config(
    genome.DefaultGenome,
    reproduction.DefaultReproduction,
    species.DefaultSpeciesSet,
    stagnation.DefaultStagnation,
    filename="config",
)


class XorFitness(IFitnessFunction):
    def __call__(self, individual: Any) -> Fitness:
        if not isinstance(individual, IndividualType):
            raise ValueError("Individual must be a numpy array")

        net = nn.FeedForwardNetwork.create(individual, neat_config)

        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outputs = [0, 1, 1, 0]

        error = sum(
            abs(net.activate(inputs[i])[0] - outputs[i]) for i in range(len(inputs))
        )
        error /= len(inputs)

        error2 = error + (len(net.node_evals) + len(net.values)) / 100

        return Fitness(
            features_fitness=[error],
            fitness=error2,
        )


class IndividualGenerator(IIndividualGenerator):
    def create_invividual(self):
        gen = genome.DefaultGenome(int(uuid.uuid4()))
        gen.configure_new(neat_config.genome_config)
        return gen

    def create_individuals(self, count: int):
        return [self.create_invividual() for _ in range(count)]

    def get_mutated_individual(self, parent1, parent2):
        gen = genome.DefaultGenome(int(uuid.uuid4()))
        parent1.fitness, parent2.fitness = 0, 0
        gen.configure_crossover(parent1, parent2, neat_config.genome_config)
        gen.mutate(neat_config.genome_config)

        return gen

    def get_mutated_individuals(
        self, individual_selector: IIndividualSelector, count: int
    ) -> Any:
        return [
            self.get_mutated_individual(
                individual_selector.get(), individual_selector.get()
            )
            for _ in range(count)
        ]


arch = GridArchive(
    dimensions=[32],
    ranges=[(-1, 1)],
    comparator=MinComparator(),
)

map_elites = MapElites(
    archive=arch,
    fitness_funcion=XorFitness(),
    individual_selector=RandomIndividualSelector(arch),
    individual_generator=IndividualGenerator(),
    initial_size=100,
)


map_elites.run(iterations=256, batch_size=512)

best = map_elites.best_solution

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [0, 1, 1, 0]

net = nn.FeedForwardNetwork.create(best, neat_config)

for i in range(len(inputs)):
    print(f"Input: {inputs[i]}")
    print(f"Output: {net.activate(inputs[i])[0]:.2f}")
    print(f"Expected: {outputs[i]}")
    print("")

print(f"Best individual: {best}")
print(net)
