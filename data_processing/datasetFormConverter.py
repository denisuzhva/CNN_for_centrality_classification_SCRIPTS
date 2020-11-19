import numpy as np



def remap3D6x6(ev_mat, in_depth):
    """Map of NA61 PSD modules to 3D np.array."""
    ev_tensor = np.zeros((6, 6, in_depth), dtype=np.float32)
    mat_range = range(17, 19)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[0][doubled_iter + 1 - 34] = ev_mat[mat_iter] / 2
        ev_tensor[0][doubled_iter + 2 - 34] = ev_mat[mat_iter] / 2
    mat_range = range(20, 22)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[doubled_iter + 1 - 40][5] = ev_mat[mat_iter] / 2
        ev_tensor[doubled_iter + 2 - 40][5] = ev_mat[mat_iter] / 2
    mat_range = range(23, 25)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 23:
            psd_iter = 24
        if mat_iter == 24:
            psd_iter = 23
        ev_tensor[5][doubled_iter + 1 - 46] = ev_mat[psd_iter] / 2
        ev_tensor[5][doubled_iter + 2 - 46] = ev_mat[psd_iter] / 2
    mat_range = range(26, 28)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 26:
            psd_iter = 27
        if mat_iter == 27:
            psd_iter = 26
        ev_tensor[doubled_iter + 1 - 52][0] = ev_mat[psd_iter] / 2
        ev_tensor[doubled_iter + 2 - 52][0] = ev_mat[psd_iter] / 2
    mat_range = range(0, 4)
    for mat_i in mat_range:
        for mat_j in mat_range:
            ev_tensor[1 + mat_i][1 + mat_j] = ev_mat[mat_j + mat_i * 4]
    return ev_tensor


def remapCenPer(ev_mat, in_depth):
    """Separate the central modules from the peripheral ones."""
    cen_tensor = np.zeros((4, 4, in_depth), dtype=np.float32)
    per_tensor = np.zeros_like(cen_tensor)
    for i in range(4):
        for j in range(4):
            cen_tensor[i, j] = ev_mat[4*i + j]
    per_tensor[0, 1] = ev_mat[17]
    per_tensor[0, 2] = ev_mat[18]
    per_tensor[1, 0] = ev_mat[27]
    per_tensor[2, 0] = ev_mat[26]
    per_tensor[1, 3] = ev_mat[20]
    per_tensor[2, 3] = ev_mat[21]
    per_tensor[3, 1] = ev_mat[24]
    per_tensor[3, 2] = ev_mat[23]
    return cen_tensor, per_tensor


if __name__ == '__main__':
    DATASET = '../../DATASETS/NA61_EPOS3_FIXprof/'
    ev_mat = np.load(DATASET + 'ev_mat.npy')
    psd_depth = ev_mat.shape[-1]
    n_events = ev_mat.shape[0]

    ev_tensor = np.zeros((n_events, 6, 6, psd_depth, 1))
    for i in range(n_events):
        ev_tensor[i, :, :, :, 0] = remap3D6x6(ev_mat[i], psd_depth)
    np.save(DATASET + 'features_6x6x10.npy', ev_tensor)
