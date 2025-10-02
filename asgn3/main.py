from pathlib import Path
from typing import Any
from multiprocessing import Pool

import mujoco
from mujoco import viewer
from networkx import DiGraph
import numpy as np

from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

from rng import RNG
from robots import (
    TrainingBrain,
    RandomRobotBody,
    Robot,
    RobotBody,
    TestBrain,
    random_body_genotype,
)

from rich.traceback import install

install(width=180, locals_max_length=100)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


class EvolutionaryAlgorithm:
    def __init__(self) -> None:
        self.processes = 8
        self.num_modules = 20
        self.genotype_size = 64
        self.body_generations = 256
        self.body_population_size = 64
        self.brain_generations = 256
        self.brain_population_size = 64

    def evolve_brains(self, robot_body: RobotBody) -> "Robot":
        input_size, output_size = self.get_input_output_sizes(robot_body)
        brains = [TrainingBrain(input_size, output_size).random() for _ in range(self.brain_population_size)]
        
        best_brain = None
        fitness = np.zeros((self.body_generations, self.body_population_size))

        for generation in range(self.brain_generations):
            pairs = []
            for brain in brains:
                robot = Robot(robot_body, brain)
                self.experiment(robot=robot, mode="simple")
                pairs.append((brain, robot.fitness(), robot_body))
            pairs.sort(key=lambda x: x[1], reverse=True)
            best_brain = pairs[0]

            fitness[generation, :] = [pair[1] for pair in pairs]

            scaled_fitnesses = np.array(
                [pair[1] - pairs[-1][1] for pair in pairs]
            )
            scaled_fitnesses /= sum(scaled_fitnesses)

            next_gen = []
            for _ in range(round(len(brains) / 4)):
                choice = RNG.choice(a=pairs, size=2, replace=False, p=scaled_fitnesses)
                p1 = choice[0][0]
                p2 = choice[1][0]
                c1, c2 = p1.crossover(p2)
                c1.mutation()
                c2.mutation()
                next_gen.append(c1)
                next_gen.append(c2)
            
            next_gen.extend([c for c in brains[: len(brains) // 2]])
            brains = next_gen
        return best_brain

    def run_random(self):
        body_genotypes = [random_body_genotype(self.genotype_size) for _ in range(self.body_population_size)]
        robot_bodies = [RandomRobotBody(body_genotype, self.num_modules) for body_genotype in body_genotypes]

        fitness = np.zeros((self.body_generations, self.body_population_size))
        best_robot = None
        for generation in range(self.body_generations):
            with Pool(processes=self.processes) as pool:
                robots = pool.map(self.evolve_brains, robot_bodies)
                robots.sort(key=lambda x: x[1], reverse=True)
                best_robot = robots[0]

                fitness[generation, :] = [r[1] for r in robots]

                scaled_fitnesses = np.array(
                    [c[1] - robots[-1][1] for c in robots]
                )
                scaled_fitnesses /= sum(scaled_fitnesses)

                next_gen = []
                for _ in range(round(len(robots) / 4)):
                    choice = RNG.choice(a=robots, size=2, replace=False, p=scaled_fitnesses)
                    p1 = choice[0][2]
                    p2 = choice[1][2]
                    c1, c2 = p1.crossover(p2)
                    c1.mutation()
                    c2.mutation()
                    next_gen.append(c1)
                    next_gen.append(c2)
                
                next_gen.extend([c for c in robots[: len(robots) // 2]])
                robots = next_gen

        return best_robot

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
                simple_runner(
                    model,
                    data,
                    duration,
                )
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


def main():
    ea = EvolutionaryAlgorithm()
    ea.run_random()


if __name__ == "__main__":
    main()
