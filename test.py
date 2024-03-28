import numpy as np
from matplotlib import pyplot as plt
from ribs.archives import GridArchive  # type: ignore
from ribs.emitters import EvolutionStrategyEmitter  # type: ignore
from ribs.schedulers import Scheduler  # type: ignore
from ribs.visualize import grid_archive_heatmap  # type: ignore

N = 4
W = 776600
weights_n = [186910, 109503, 71040, 86400]
prices_n = [1.0, 1.0, 1.0, 1.0]
count_limits = [30, 30, 30, 30]

weights = np.array(weights_n)
prices = np.array(prices_n)


archive = GridArchive(
    solution_dim=N,
    dims=[32, 32],
    ranges=[(0, 100), (0, max(np.sum(prices * count_limits), 0.1))],  # type: ignore
)


emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=[0.0] * N,
        sigma0=0.1,
    )
    for _ in range(3)
]


def calc_coverage(solution):
    # Calculate the coverage of the solution.
    sum_weights = np.sum(np.abs(np.around(solution * count_limits)) * weights)
    sum_prices = np.sum(np.abs(np.around(solution * count_limits)) * prices)
    return sum_weights, sum_prices


def calc_coverage_V2(solutions: np.ndarray):
    # Calculate the coverage of the solutions.
    sum_weights = 100 / (
        np.abs(
            W - np.sum(np.abs(np.around(solutions * count_limits)) * weights, axis=1)
        )
        + 0.000001
    )
    sum_prices = np.sum(np.abs(np.around(solutions * count_limits)) * prices, axis=1)
    results = np.column_stack((sum_weights, sum_prices))
    return sum_weights, results


scheduler = Scheduler(archive, emitters)
import numpy as np

for _ in range(512):
    solutions = scheduler.ask()

    obj_path, results = calc_coverage_V2(solutions)

    scheduler.tell(obj_path, results)


best = archive.best_elite

print(best)
solutions = np.array(best["solution"]) * count_limits
print(np.abs(np.around(solutions)))
print(np.abs(np.around(solutions)) * weights)
print(np.sum(np.abs(np.around(solutions)) * weights))
print("00000000")
print("Results", calc_coverage(np.array([best["solution"]])))
print("Results2", calc_coverage_V2(np.array([best["solution"]])))

best2 = best
print("00000000")
for elite in archive:
    if elite["measures"][1] > best2["measures"][1] and W >= np.sum(
        (np.abs(np.around(elite["solution"])) * weights)
    ):
        best2 = elite

print(best2)
best = best2
solutions = np.array(best["solution"]) * count_limits
print(np.abs(np.around(solutions)))
print(np.abs(np.around(solutions)) * weights)
print(np.sum(np.abs(np.around(solutions)) * weights))
print("00000000")
print("Results", calc_coverage(np.array([best["solution"]])))
print("Results2", calc_coverage_V2(np.array([best["solution"]])))
