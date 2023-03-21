from .network import DensityNetwork


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    else:
        raise NotImplementedError("Unknown network typeß!")

