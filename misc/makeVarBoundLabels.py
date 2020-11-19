import numpy as np



DATASET = '../../DATASETS/EPOS_EV2prof/'
nrg = np.load(DATASET + 'nrg24.npy')
spec = np.load(DATASET + 'spec24.npy')
n_of_events = spec.size

spec_bounds = np.arange(7)

nrg_sorted = np.sort(nrg)
lab_spec = np.zeros((n_of_events, 2), dtype=np.float32)
lab_nrg = np.zeros_like(lab_spec)

for bound in spec_bounds:
    perc = spec[spec <= bound].size / spec.size
    nrg_bound = nrg_sorted[int(perc * nrg.size)]
    lab_spec_supp = np.zeros_like(spec, dtype=int)
    lab_spec_supp[spec > bound] = 1
    lab_spec[:, 1] = lab_spec_supp
    lab_spec[:, 0] = 1-lab_spec_supp
    lab_nrg = np.zeros_like(lab_spec)
    lab_nrg_supp = np.zeros_like(nrg, dtype=int)
    lab_nrg_supp[nrg > nrg_bound] = 1
    lab_nrg[:, 1] = lab_nrg_supp
    lab_nrg[:, 0] = 1-lab_nrg_supp

    np.save(DATASET + f'labels_spec{bound}.npy', lab_spec)
    np.save(DATASET + f'labels_nrg{bound}.npy', lab_nrg)

