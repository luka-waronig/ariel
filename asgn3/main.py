from pathlib import Path
from typing import Any
from copy import deepcopy

import mujoco
from mujoco import viewer
from networkx import DiGraph
import numpy as np
from numpy.typing import NDArray

from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

from rng import RNG
from robots import (
    RandomBrain,
    RandomRobotBody,
    Robot,
    RobotBody,
    TestBrain,
    random_body_genotype,
)


SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


class EvolutionaryAlgorithm:
    def __init__(self) -> None:
        self.num_modules = 20
        self.genotype_size = 64

    def run_random(self):
        body_genotype = random_body_genotype(self.genotype_size)
        robot_body = RandomRobotBody(body_genotype, self.num_modules)

        input_size, output_size = self.get_input_output_sizes(robot_body)
        robot = Robot(robot_body, RandomBrain(input_size, output_size))

        self.experiment(robot=robot, mode="launcher")

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
                    duration=0.1,
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
