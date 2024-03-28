import uuid

# import freeze_support for windows
from multiprocessing import freeze_support
from typing import Any

import neat
import numpy as np
from neat import reproduction  # type: ignore
from neat import config, genome, nn, population, species, stagnation

from mapelites import *
from mapelites.interfaces import Fitness, IIndividualSelector

IndividualType = np.ndarray

parameters = [
    "compatibility_disjoint_coefficient",
    "compatibility_weight_coefficient",
    "node_add_prob",
    "node_delete_prob",
    "conn_add_prob",
    "conn_delete_prob",
]

parameters_ranges = [
    (0.0, 3.0),
    (0.0, 3.0),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
]


xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config):
    # Load configuration.

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    # save_fitness_reporter = SaveFitnessReporter()
    # p.add_reporter(save_fitness_reporter)
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)

    winner = p.run(eval_genomes, 16)

    eval_genomes([(0, winner)], config)

    return winner.fitness


class XorFitness(IFitnessFunction):
    def __call__(self, individual: Any) -> Fitness:
        neat_config = config.Config(
            genome.DefaultGenome,
            reproduction.DefaultReproduction,
            species.DefaultSpeciesSet,
            stagnation.DefaultStagnation,
            filename="neat-xor-benchmark-config",
        )

        for parameter, value in zip(parameters, individual):
            setattr(neat_config.genome_config, parameter, value)

        runs_count = 8
        fitness = 0.0
        print(f"Evaluating of individual: {individual} started")
        for _ in range(runs_count):
            fitness += run(neat_config)
        print(f"Evaluating of individual: {individual} finished")

        fitness /= runs_count

        return Fitness(
            features_fitness=[
                individual[0],
                individual[1],
            ],
            fitness=fitness,
        )


def mut_gaussian(
    individual: Any, mu: float = 0.0, sigma: float = 1.0, mut_pb: float = 0.2
):
    for i in range(len(individual)):
        if random.random() < mut_pb:
            individual[i] += random.gauss(mu, sigma)
    for i, i_range in enumerate(parameters_ranges):
        individual[i] = max(i_range[0], min(i_range[1], individual[i]))

    return individual


class IndividualGenerator(IIndividualGenerator):
    def create_invividual(self):
        return np.array(np.random.rand(len(parameters)))

    def create_individuals(self, count: int):
        return [self.create_invividual() for _ in range(count)]

    def get_mutated_individuals(
        self, individual_selector: IIndividualSelector, count: int
    ) -> Any:
        return [mut_gaussian(individual_selector.get()) for _ in range(count)]


def global_run():
    arch = GridArchive(
        dimensions=[16, 16],
        ranges=[(0.0, 3.0), (0.0, 3.0)],
        comparator=MaxComparator(),
    )

    map_elites = MapElites(
        archive=arch,
        fitness_funcion=XorFitness(),
        individual_selector=RandomIndividualSelector(arch),
        individual_generator=IndividualGenerator(),
        fitness_comparator=MaxComparator(),
        initial_size=32,
    )

    fitness_history = map_elites.run(iterations=32, batch_size=8, verbose=True)

    best_solution = map_elites.best_solution

    print(f"Best solution: {best_solution}")
    print(f"Best solution fitness: {arch.fitness}")

    for parameter, value in zip(parameters, best_solution):
        print(f"{parameter}: {value}")


if __name__ == "__main__":
    freeze_support()

    tests_count = 1024
    average_error = 0.0
    average_time = 0.0

    average_fitness = [0.0] * 128

    global_run()

    # for _ in range(tests_count):
    #     start = time.time()
    #     fitnesses, error = run()
    #     end = time.time()

    #     for index in range(128):
    #         average_fitness[index] += fitnesses[index]

    #     average_time += end - start
    #     average_error += error

    # average_error /= tests_count
    # average_time /= tests_count

    # for index in range(128):
    #     average_fitness[index] /= tests_count

    # print(f"Average error: {average_error}")
    # print(f"Average time: {average_time}")

    # import json

    # with open("neat_mp_xor_benchmark_average_fitness.txt", "w") as file:
    #     json.dump(average_fitness, file)
