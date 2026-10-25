import math

def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    def get_dim(space: dict) -> int:
        stype = space.get('type')
        if stype == 'Box':
            return math.prod(space['shape'])
        elif stype in ('Discrete', 'MultiBinary'):
            return space['n']
        raise ValueError(f"Unsupported space type: {stype}")

    input_dim = get_dim(obs_space)
    output_dim = get_dim(act_space)

    dims = [input_dim] + list(hidden_layers) + [output_dim]
    
    layer_shapes = []
    total_params = 0

    for i in range(len(dims) - 1):
        in_f = dims[i]
        out_f = dims[i + 1]
        layer_shapes.append(((in_f, out_f), (out_f,)))
        total_params += (in_f * out_f) + out_f

    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'layer_shapes': layer_shapes,
        'total_params': total_params
    }