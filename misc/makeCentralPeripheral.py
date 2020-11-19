import numpy as np
import matplotlib.pyplot as plt



def separateParts(dataset_nonflat):
    dataset_central = dataset_nonflat[:, 2:6, 2:6, :]
    dataset_peripheral = np.zeros_like(dataset_central)
    dataset_peripheral[:, 0, 1, :] = dataset_nonflat[:, 1, 3, :]
    dataset_peripheral[:, 0, 2, :] = dataset_nonflat[:, 1, 5, :]
    dataset_peripheral[:, 1, 0, :] = dataset_nonflat[:, 3, 1, :]
    dataset_peripheral[:, 1, 3, :] = dataset_nonflat[:, 3, 7, :]
    dataset_peripheral[:, 2, 0, :] = dataset_nonflat[:, 5, 1, :]
    dataset_peripheral[:, 2, 3, :] = dataset_nonflat[:, 5, 7, :]
    dataset_peripheral[:, 3, 1, :] = dataset_nonflat[:, 7, 3, :]
    dataset_peripheral[:, 3, 2, :] = dataset_nonflat[:, 7, 5, :]

    return dataset_central, dataset_peripheral


if __name__ == '__main__':
    all_dataset = np.load('../../DATASETS/NA61_better10_8x8/all8/all_dataset_flat.npy')
    all_dataset_nonflat = all_dataset.reshape(-1, 8, 8, 10)
    #all_dataset_nonflat = np.swapaxes(all_dataset_nonflat, 1, 3)
   
    dataset_central, dataset_peripheral = separateParts(all_dataset_nonflat)
    dataset_central_flat = dataset_central.reshape(-1, 160)
    dataset_peripheral_flat = dataset_peripheral.reshape(-1, 160)

    np.save('../../DATASETS/NA61_better10_8x8/all8/all_dataset_central_flat.npy', dataset_central_flat)
    np.save('../../DATASETS/NA61_better10_8x8/all8/all_dataset_peripheral_flat.npy', dataset_peripheral_flat)
