from .hashencoder import HashEncoder
from .freqencoder import FreqEncoder


def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19,
                **kwargs):

    if encoding == "None":
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == "frequency":
        encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    elif encoding == "hashgrid":
        encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
            base_resolution=base_resolution, 
            log2_hashmap_size=log2_hashmap_size)

    else:
        raise NotImplementedError()

    return encoder