from typing import Any
import mujoco
from networkx import DiGraph
import numpy as np
from numpy.typing import NDArray

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

from rng import RNG


class RobotBody:
    def __init__(self, body_genotype, num_modules: int) -> None:
        pass


class RandomRobotBody(RobotBody):
    def __init__(
        self, body_genotype: list[NDArray[np.float32]], num_modules: int
    ) -> None:
        super().__init__(body_genotype, num_modules)
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        p_matrices = nde.forward(body_genotype)

        hpd = HighProbabilityDecoder(num_modules)
        self.robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )


class Brain:
    def __init__(self, input_size: int, output_size: int) -> None:
        pass

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Any:
        pass


class TestBrain(Brain):
    """Class to determine input and output layer sizes for robot controllers."""

    def __init__(self) -> None:
        self.input_size = None
        self.output_size = None

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        self.input_size = len(data.qpos)
        self.output_size = model.nu
        return super().__call__(model, data)


class RandomBrain(Brain):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        hidden_size = 8

        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        self.w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
        self.w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
        self.w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> Any:
        # Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.w1))
        layer2 = np.tanh(np.dot(layer1, self.w2))
        outputs = np.tanh(np.dot(layer2, self.w3))

        # Scale the outputs
        return outputs * np.pi


class Robot:
    """
    A combination of a RobotBody and a Brain. Robots can only be used once in
    a MuJoCo simulation.
    """

    def __init__(self, body: RobotBody, brain: Brain) -> None:
        self.core = construct_mjspec_from_graph(body.robot_graph)
        self.controller = Controller(
            controller_callback_function=brain,
            tracker=self.get_tracker(),
        )

    def get_tracker(self):
        mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )
        return tracker


def random_body_genotype(genotype_size: int):
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    return genotype
