from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.rcParams["figure.constrained_layout.use"] = True


class LivePlotter:
    def __init__(self, fitness: NDArray[np.float32], directory: Path) -> None:
        self.fitness = fitness
        self.directory = directory

    def plot(self) -> None:
        averages = np.average(self.fitness, axis=1)
        quantiles = np.quantile(self.fitness, np.linspace(0, 100, 11), axis=1)
        current_gen = max(np.nonzero(averages)) - 1
        gen_range = np.arange(current_gen + 1)

        plt.plot(gen_range, averages, "r-", label="average")
        for q in range(10):
            plt.plot(gen_range, quantiles[q], "k-")
        plt.plot(gen_range, quantiles[11], "k-", label="quantiles")

        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title(f"Progress generation {current_gen}")
        plt.legend()
        plt.savefig(self.directory.joinpath("fitness_plot.pdf"))
        plt.close()
