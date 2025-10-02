from pathlib import Path
from typing import Any
from collections.abc import Sequence
from multiprocessing import Pool

import mujoco
from mujoco import viewer
import numpy as np
from numpy.typing import NDArray

from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

from rng import NP_RNG, RNG
from robots import (
    Brain,
    TrainingBrain,
    RandomRobotBody,
    Robot,
    RobotBody,
    TestBrain,
    random_body_genotype,
)

from rich.traceback import install

install(width=180, show_locals=False)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


class EvolutionaryAlgorithm:
    def __init__(self) -> None:
        self.processes = 8
        self.num_modules = 20
        self.genotype_size = 64
        # self.body_generations = 256
        # self.body_population_size = 64
        # self.brain_generations = 256
        # self.brain_population_size = 64
        self.body_generations = 10
        self.body_population_size = 8
        self.brain_generations = 1
        self.brain_population_size = 8

        assert self.brain_population_size % 4 == 0, "Populations must be div. by 4."
        assert self.body_population_size % 4 == 0, "Populations must be div. by 4."

    def evolve_brains(self, robot_body: RobotBody) -> tuple[Brain, float]:
        # The bodies get fresh new brains at the start of learning
        brains = self.generate_brains(robot_body)

        best_brain: tuple[Brain, float]
        fitness = np.zeros((self.body_generations, self.body_population_size))

        for generation in range(self.brain_generations):
            brains_fitness: list[tuple[Brain, float]] = []
            for brain in brains:
                robot = Robot(robot_body, brain)
                self.experiment(robot=robot, mode="simple")
                brains_fitness.append((brain, robot.fitness()))
            brains_fitness.sort(key=fitness_key, reverse=True)
            best_brain = brains_fitness[0]

            fitness[generation, :] = [pair[1] for pair in brains_fitness]

            scaled_fitnesses = np.array(
                [pair[1] - brains_fitness[-1][1] for pair in brains_fitness]
            )
            scaled_fitnesses /= sum(scaled_fitnesses)

            next_gen = self.children_brains(brains_fitness, scaled_fitnesses)
            brains = next_gen

            # solves a type hinting problem
            if generation == self.brain_generations - 1:
                return best_brain
        raise ValueError("self.brain_generations must be at least 1.")

    def run_random(self):
        # Create body population
        robot_bodies = self.generate_bodies()

        fitness = np.zeros((self.body_generations, self.body_population_size))
        best_robot = None

        for generation in range(self.body_generations):
            # Use multiprocessing to speed up computations
            with Pool(processes=self.processes) as pool:
                bodies_fitness = pool.map(self.evolve_brains, robot_bodies)

            bodies_fitness.sort(key=fitness_key, reverse=True)
            best_robot = bodies_fitness[0]

            fitness[generation, :] = [r[1] for r in bodies_fitness]
            scaled_fitnesses = np.array(
                [c[1] - bodies_fitness[-1][1] for c in bodies_fitness]
            )
            scaled_fitnesses /= sum(scaled_fitnesses)

            next_gen = self.children_bodies(bodies_fitness, scaled_fitnesses)
            robot_bodies = next_gen

        return best_robot

    def children_brains(
        self,
        brains_fitness: list[tuple[Brain, float]],
        scaled_fitnesses: NDArray[np.float32],
    ) -> list[Brain]:
        next_gen = []
        for _ in range(round(len(brains_fitness) / 4)):
            choice = RNG.choices(brains_fitness, weights=scaled_fitnesses, k=2)
            p1 = choice[0][0]
            p2 = choice[1][0]
            c1, c2 = p1.crossover(p2)
            c1.mutation()
            c2.mutation()
            next_gen.append(c1)
            next_gen.append(c2)

        next_gen.extend(
            [c[0].copy() for c in brains_fitness[: len(brains_fitness) // 2]]
        )
        return next_gen

    def children_bodies(
        self,
        bodies_fitness: list[tuple[RobotBody, float]],
        scaled_fitnesses: NDArray[np.float32],
    ) -> list[RobotBody]:
        next_gen = []
        for _ in range(round(len(bodies_fitness) / 4)):
            choice = RNG.choices(bodies_fitness, weights=scaled_fitnesses, k=2)

            p1: RobotBody = choice[0][0]
            p2: RobotBody = choice[1][0]
            c1, c2 = p1.crossover(p2)
            c1.mutation()
            c2.mutation()
            next_gen.append(c1)
            next_gen.append(c2)

        next_gen.extend([c[0] for c in bodies_fitness[: len(bodies_fitness) // 2]])
        return next_gen

    def generate_brains(self, robot_body: RobotBody) -> Sequence[Brain]:
        input_size, output_size = self.get_input_output_sizes(robot_body)
        brains = [
            TrainingBrain(input_size, output_size).random()
            for _ in range(self.brain_population_size)
        ]

        return brains

    def generate_bodies(self):
        body_genotypes = [
            random_body_genotype(self.genotype_size)
            for _ in range(self.body_population_size)
        ]
        robot_bodies = [
            RandomRobotBody(body_genotype, self.num_modules)
            for body_genotype in body_genotypes
        ]

        return robot_bodies

    def experiment(
        self,
        robot: Robot,
        duration: int = 15,
        mode: str = "viewer",
    ) -> None:
        """Run the simulation with random movements."""
        # ==================================================================== #
        # Initialise controller to controller to None, always in the beginning.
        mujoco.set_mjcb_control(None)  # DO NOT REMOVE

        world, model, data = self.compile_world(robot)

        # Pass the model and data to the tracker
        if robot.controller.tracker is not None:
            robot.controller.tracker.setup(world.spec, data)

        # Set the control callback function
        # This is called every time step to get the next action.
        args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
        kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

        mujoco.set_mjcb_control(
            lambda m, d: robot.controller.set_control(m, d, *args, **kwargs),  # type: ignore
        )

        # ------------------------------------------------------------------ #
        match mode:  # type: ignore
            case "simple":
                # This disables visualisation (fastest option)
                simple_runner(model, data, duration, steps_per_loop=500)
            case "frame":
                # Render a single frame (for debugging)
                save_path = str(DATA / "robot.png")
                single_frame_renderer(model, data, save=True, save_path=save_path)
            case "video":
                # This records a video of the simulation
                path_to_video_folder = str(DATA / "videos")
                video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # Render with video recorder
                video_renderer(
                    model,
                    data,
                    duration=duration,
                    video_recorder=video_recorder,
                )
            case "launcher":
                # This opens a liver viewer of the simulation
                viewer.launch(
                    model=model,
                    data=data,
                )
            case "no_control":
                # If mujoco.set_mjcb_control(None), you can control the limbs manually.
                mujoco.set_mjcb_control(None)
                viewer.launch(
                    model=model,
                    data=data,
                )

    def compile_world(self, robot: Robot):
        world = OlympicArena()

        # Spawn robot in the world
        # Check docstring for spawn conditions
        world.spawn(robot.core.spec, spawn_position=[0, 0, 0.1])

        # Generate the model and data
        # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)
        return world, model, data

    def get_input_output_sizes(self, robot_body: RobotBody) -> tuple[int, int]:
        """
        Create a MuJoCo world to determine the needed sizes of the input and
        output layers for a given robot body. Try to only run this once per body.

        :param self:
        :param robot_body: The body to determine the layer sizes for.
        :type robot_body: RobotBody
        :return: The input and output sizes
        :rtype: tuple[int, int]
        """

        robot = Robot(robot_body, TestBrain())
        _, model, data = self.compile_world(robot)

        input_size = len(data.qpos)
        output_size = model.nu
        return input_size, output_size


def fitness_key(fitness_tuple: tuple[Any, float]) -> float:
    return fitness_tuple[1]


def main():
    ea = EvolutionaryAlgorithm()
    ea.run_random()


if __name__ == "__main__":
    main()
