import numpy as np



spec = np.load('../../DATASETS/NA61_SHIELD/all/spec24.npy')
nrg = np.load('../../DATASETS/NA61_SHIELD/all/nrg24.npy')

perc = spec[spec <= 3].size / spec.size
#print(perc)
nrg_sorted = np.sort(nrg)
nrg_bound = nrg_sorted[int(perc * nrg.size)]
#print(nrg_bound)

lab_spec = np.zeros((spec.size, 2))
lab_spec_supp = np.zeros_like(spec, dtype=int)
lab_spec_supp[spec > 3] = 1
lab_spec[:, 1] = lab_spec_supp
lab_spec[:, 0] = 1-lab_spec_supp
lab_nrg = np.zeros_like(lab_spec)
lab_nrg_supp = np.zeros_like(nrg, dtype=int)
lab_nrg_supp[nrg > nrg_bound] = 1
lab_nrg[:, 1] = lab_nrg_supp
lab_nrg[:, 0] = 1-lab_nrg_supp

np.save('../../DATASETS/NA61_SHIELD/all/labels_spec.npy', lab_spec)
np.save('../../DATASETS/NA61_SHIELD/all/labels_nrg.npy', lab_nrg)