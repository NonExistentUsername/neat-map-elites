import json

import matplotlib.pyplot as plt

# Load the average fitness data
name1 = "neat_mp_xor_benchmark_average_fitness.txt"
name2 = "neat_xor_benchmark_average_fitness.txt"


with open(name1, "r") as file:
    points1 = json.load(file)

with open(name2, "r") as file:
    points2 = json.load(file)

print(points1[60] - points2[60])

# Plot the average fitness
plt.plot(points1, label="Map Elites + NEAT")
plt.plot(points2, label="NEAT")

# Save the plot
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.legend()

plt.savefig("average_fitness.png")
