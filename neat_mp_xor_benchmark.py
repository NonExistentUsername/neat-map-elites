import uuid

# import freeze_support for windows
from multiprocessing import freeze_support
from typing import Any

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
    filename="neat-xor-benchmark-config",
)


class XorFitness(IFitnessFunction):
    def __call__(self, individual: Any) -> Fitness:
        net = nn.FeedForwardNetwork.create(individual, neat_config)

        xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        xor_outputs = [0, 1, 1, 0]
        net_outputs = [net.activate(inputs)[0] for inputs in xor_inputs]

        error = 4 - sum(
            (net_outputs[i] - xor_outputs[i]) ** 2 for i in range(len(xor_inputs))
        )
        feature_finess_0 = sum(net_outputs) / len(net_outputs)
        feature_finess_1 = error

        return Fitness(
            features_fitness=[feature_finess_0],
            fitness=feature_finess_1,
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


def run():

    arch = GridArchive(
        dimensions=[8],
        ranges=[(0, 1)],
        comparator=MaxComparator(),
    )

    map_elites = MapElites(
        archive=arch,
        fitness_funcion=XorFitness(),
        individual_selector=RandomIndividualSelector(arch),
        individual_generator=IndividualGenerator(),
        fitness_comparator=MaxComparator(),
        initial_size=150,
    )

    fitness_history = map_elites.run(
        iterations=128, batch_size=150, verbose=False, stop_fitness=3.99
    )

    best_solution = map_elites.best_solution
    winner_net = nn.FeedForwardNetwork.create(best_solution, neat_config)
    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_outputs = [0, 1, 1, 0]
    error = 0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        error += abs(output[0] - xo)

    return fitness_history, error / len(xor_inputs)


if __name__ == "__main__":
    freeze_support()

    tests_count = 256
    average_error = 0.0
    average_time = 0.0

    # average_fitness = [0.0] * 128

    for _ in range(tests_count):
        start = time.time()
        fitnesses, error = run()
        end = time.time()

        # for index in range(128):
        #     average_fitness[index] += fitnesses[index]

        average_time += end - start
        average_error += error

    average_error /= tests_count
    average_time /= tests_count

    # for index in range(128):
    #     average_fitness[index] /= tests_count

    print(f"Average error: {average_error}")
    print(f"Average time: {average_time}")

    import json

    with open("neat_mp_xor_benchmark_average_fitness.txt", "w") as file:
        json.dump(average_fitness, file)
