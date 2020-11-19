import numpy as np 



def make_flat(dataset_old):
    dataset = np.reshape(dataset_old, [-1, 8, 8, 10])
    dataset = dataset[:, :, 1:, :]
    dataset = dataset[:, :, :-1, :]
    dataset = dataset[:, 1:, :, :]
    dataset = dataset[:, :-1, :, :]

    dataset_new = np.zeros((dataset.shape[0], 240))
    for i in list(range(4)):
        for j in list(range(4)):
            dataset_new[:, (i + 4*j)*10:(i + 4*j)*10+10] = dataset[:, i+1, j+1, :]
    dataset_new[:, 160:170] = dataset[:, 0, 1, :]
    dataset_new[:, 170:180] = dataset[:, 0, 3, :]
    dataset_new[:, 180:190] = dataset[:, 1, -1, :]
    dataset_new[:, 190:200] = dataset[:, 3, -1, :]
    dataset_new[:, 200:210] = dataset[:, -1, 3, :]
    dataset_new[:, 210:220] = dataset[:, -1, 1, :]
    dataset_new[:, 220:230] = dataset[:, 3, 0, :]
    dataset_new[:, 230:240] = dataset[:, 1, 0, :]

    return dataset_new


def clear_from_zero_mul(dataset, multiplicities):
    zeros = np.where(multiplicities==0)[0]
    dataset = np.delete(dataset, zeros, axis=0)

    return dataset


def split_test_train(dataset):
    full_train_dataset = dataset[0:80000, :]
    full_test_dataset = dataset[80000:dataset.shape[0], :]
    mid_train_dataset = dataset[0:5000, :]
    mid_test_dataset = dataset[5000:9000, :]
    
    return [full_train_dataset, 
            full_test_dataset, 
            mid_train_dataset, 
            mid_test_dataset]



if __name__ == "__main__":
    all_dataset = np.load('../../DATASETS/NA61_better10_8x8/all/all_dataset_flat.npy')
    all_dataset_sum = np.load('../../DATASETS/NA61_better10_8x8/all/all_dataset_sum.npy')
    all_labels_nrg = np.load('../../DATASETS/NA61_better10_8x8/all/all_labels_nrg_flat.npy')
    all_labels_spec = np.load('../../DATASETS/NA61_better10_8x8/all/all_labels_spec_flat.npy')
    mul_data = np.load('../../DATASETS/NA61_better10_8x8/all/all_mul.npy')

    #all_dataset_new = make_flat(all_dataset)
    
    all_dataset = clear_from_zero_mul(all_dataset, mul_data)
    all_dataset_sum = clear_from_zero_mul(all_dataset_sum, mul_data)
    all_labels_nrg = clear_from_zero_mul(all_labels_nrg, mul_data)
    all_labels_spec = clear_from_zero_mul(all_labels_spec, mul_data)

    print(all_dataset.shape)
    print(all_dataset_sum.shape)
    print(all_labels_nrg.shape)
    print(all_labels_spec.shape)

    dataset_name = 'NA61_better10_8x8_nozero'
    np.save('../../DATASETS/{}/all/all_dataset_flat.npy'.format(dataset_name), all_dataset)
    np.save('../../DATASETS/{}/all/all_dataset_sum.npy'.format(dataset_name), all_dataset_sum)
    np.save('../../DATASETS/{}/all/all_labels_nrg_flat.npy'.format(dataset_name), all_labels_nrg)
    np.save('../../DATASETS/{}/all/all_labels_spec_flat.npy'.format(dataset_name), all_labels_spec)


