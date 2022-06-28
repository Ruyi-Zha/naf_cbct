from .network import DensityNetwork

def get_network(type):
    if type == 'mlp':
        return DensityNetwork
    else:
        NotImplementedError('Unknown network typeß!')

