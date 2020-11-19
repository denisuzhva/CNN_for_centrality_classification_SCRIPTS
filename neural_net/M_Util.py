#############################
##### UTILITY FUNCTIONS #####
#############################



import numpy as np



def reduceData(dset_features, dset_labels, train_size, reduce_factor):
    dset_size = dset_labels.shape[0]
    train_size_new = train_size // reduce_factor
    valid_size = dset_size - train_size
    valid_size_new = valid_size // reduce_factor
    dset_size_new = train_size_new + valid_size_new

    for key in dset_features.keys():
        ft_tr = dset_features[key][:train_size_new]
        ft_vl = dset_features[key][train_size:(train_size+valid_size_new)]
        dset_features[key] = np.concatenate((ft_tr, ft_vl), axis=0)

    lab_tr = dset_labels[:train_size_new]
    lab_vl = dset_labels[train_size:(train_size+valid_size_new)]
    dset_labels = np.concatenate((lab_tr, lab_vl), axis=0)

    return dset_features, dset_labels, train_size_new, valid_size_new, dset_size_new


def shuffleData(dset_features, dset_labels):
    dset_size = dset_labels.shape[0]
    p = np.random.permutation(dset_size)
    for key in dset_features.keys():
        dset_features[key] = dset_features[key][p]
    dset_labels = dset_labels[p]

    return dset_features, dset_labels, p


def makeNoise(dset_features, train_size, r_disp=0.2, r_mean=1.):
    for key in dset_features.keys():
        ft = dset_features[key]
        test_size = ft.shape[0]
        ft_tensor_dim = ft.shape[1:-1]
        ft_noise = np.ones_like(ft)
        ft_noise_single = np.random.randn(1, ft_tensor_dim[0], ft_tensor_dim[1], ft_tensor_dim[2], 1) * np.sqrt(r_disp) + r_mean
        ft_noise_act = np.repeat(ft_noise_single, test_size, axis=0)
        ft_noise[train_size:, :, :, :, :] = ft_noise_act
        dset_features[key] = np.multiply(ft, ft_noise)

    return dset_features


def balanceDatasets(dset_features, dset_labels):
    lab = dset_labels.argmax(axis=1)
    smallest_class = np.argmin([np.sum(lab == 0), np.sum(lab == 1)])
    n_in_smallest_class = np.sum(lab == smallest_class)
    p = np.random.permutation(n_in_smallest_class * 2)

    new_features = {}
    for key in dset_features.keys():
        new_features[key] = dset_features[key][lab == smallest_class]
        new_features[key] = np.concatenate((new_features[key], 
            dset_features[key][lab == np.abs(smallest_class - 1)][:n_in_smallest_class]), 
            axis=0)
        new_features[key] = new_features[key][p]

    new_labels = dset_labels[lab == smallest_class]
    new_labels = np.concatenate((new_labels, dset_labels[lab == np.abs(smallest_class - 1)]))
    new_labels = new_labels[p]

    return new_features, new_labels