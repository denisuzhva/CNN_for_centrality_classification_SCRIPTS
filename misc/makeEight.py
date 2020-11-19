import numpy as np 


if __name__ == "__main__":
    all_dataset = np.load('../../DATASETS/NA61_better10_8x8/all/all_dataset_flat.npy')
    all_sum = np.load('../../DATASETS/NA61_better10_8x8/all/all_dataset_sum.npy')
    all_lab_nrg = np.load('../../DATASETS/NA61_better10_8x8/all/all_labels_nrg.npy')
    all_lab_spec = np.load('../../DATASETS/NA61_better10_8x8/all/all_labels_spec.npy')
    all_mul = np.load('../../DATASETS/NA61_better10_8x8/all/all_mul.npy')

    all_dataset = all_dataset[0:80000, :]
    all_sum = all_sum[0:80000, :]
    all_lab_nrg = all_lab_nrg[0:80000, :]
    all_lab_spec = all_lab_spec[0:80000, :]
    all_mul = all_mul[0:80000]

    np.save('../../DATASETS/NA61_better10_8x8/all8/all_dataset_flat.npy', all_dataset)
    np.save('../../DATASETS/NA61_better10_8x8/all8/all_dataset_sum.npy', all_sum)
    np.save('../../DATASETS/NA61_better10_8x8/all8/all_labels_nrg.npy', all_lab_nrg)
    np.save('../../DATASETS/NA61_better10_8x8/all8/all_labels_spec.npy', all_lab_spec)
    np.save('../../DATASETS/NA61_better10_8x8/all8/all_mul.npy', all_mul)
