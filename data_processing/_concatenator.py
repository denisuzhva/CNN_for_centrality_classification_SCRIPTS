########################
# Concatenate test and #
# train .npy datasets  #
########################



import numpy as np


train_dataset = np.load('../AI/NA61_minicross10_8x8/full/train_dataset_flat.npy')
train_labels_nrg = np.load('../AI/NA61_minicross10_8x8/full/train_labels_nrg_flat.npy')
train_labels_spec = np.load('../AI/NA61_minicross10_8x8/full/train_labels_spec_flat.npy')
test_dataset = np.load('../AI/NA61_minicross10_8x8/full/test_dataset_flat.npy')
test_labels_nrg = np.load('../AI/NA61_minicross10_8x8/full/test_labels_nrg_flat.npy')
test_labels_spec = np.load('../AI/NA61_minicross10_8x8/full/test_labels_spec_flat.npy')

all_dataset = np.concatenate((train_dataset, test_dataset), axis=0)
all_labels_nrg = np.concatenate((train_labels_nrg, test_labels_nrg), axis=0)
all_labels_spec = np.concatenate((train_labels_spec, test_labels_spec), axis=0)

#print(all_labels_spec.shape)

np.save('../AI/NA61_minicross10_8x8/all/all_dataset_flat.npy', all_dataset)
np.save('../AI/NA61_minicross10_8x8/all/all_labels_nrg_flat.npy', all_labels_nrg)
np.save('../AI/NA61_minicross10_8x8/all/all_labels_spec_flat.npy', all_labels_spec)
