"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os
import time

import neat

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


class SaveFitnessReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.fitness = []

    def post_evaluate(self, config, population, species, best_genome):
        self.fitness.append(best_genome.fitness)


def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    save_fitness_reporter = SaveFitnessReporter()
    p.add_reporter(save_fitness_reporter)
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 128)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    error = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        error += abs(output[0] - xo[0])

    return save_fitness_reporter.fitness, error / len(xor_inputs)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-xor-benchmark-config")

    tests_count = 256
    average_error = 0.0
    average_time = 0.0

    # average_fitness = [0.0] * 128

    for _ in range(tests_count):
        start = time.time()
        fitnesses, error = run(config_path)
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

    # import json

    # with open("neat_xor_benchmark_average_fitness.txt", "w") as file:
    #     json.dump(average_fitness, file)
