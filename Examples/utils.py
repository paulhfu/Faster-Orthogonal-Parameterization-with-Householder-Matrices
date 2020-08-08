import numpy as np

def ind_flat_2_spat(flat_indices, shape):
    spat_indices = np.zeros([len(flat_indices)] + [len(shape)], dtype=np.integer)
    for flat_ind, spat_ind in zip(flat_indices, spat_indices):
        rm = flat_ind
        for dim in range(1, len(shape)):
            sz = np.prod(shape[dim:])
            spat_ind[dim - 1] = rm // sz
            rm -= spat_ind[dim - 1] * sz
        spat_ind[-1] = rm
    return spat_indices


def ind_spat_2_flat(spat_indices, shape):
    flat_indices = np.zeros(len(spat_indices), dtype=np.integer)
    for i, spat_ind in enumerate(spat_indices):
        for dim in range(len(shape)):
            flat_indices[i] += max(1, np.prod(shape[dim + 1:])) * spat_ind[dim]
    return flat_indices