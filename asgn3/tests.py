import json
from main import EvolutionaryAlgorithm, export_nde, import_nde
from robots import RandomRobotBody


class TestEA(EvolutionaryAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.body_population_size = 1


def saving_nde():
    ea1 = TestEA()
    body1: RandomRobotBody = ea1.generate_bodies()[0]  # type: ignore
    genotype = body1.genotype

    nde_string = json.dumps(export_nde(ea1.nde))
    nde = import_nde(json.loads(nde_string))

    body2 = RandomRobotBody(genotype, ea1.num_modules, nde)

    for node in body2.robot_graph.nodes:
        assert body1.robot_graph.has_node(
            node
        ), f"\n{body1.robot_graph.edges}\n{body2.robot_graph.edges}"
    for edge in body2.robot_graph.edges:
        assert body1.robot_graph.has_edge(
            *edge
        ), f"\n{body1.robot_graph.edges}\n{body2.robot_graph.edges}"


if __name__ == "__main__":
    saving_nde()
