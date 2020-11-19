import numpy as np



nrg_sh = np.load('../../DATASETS/NA61_SHIELD/all/nrg24.npy')
nrg_ep = np.load('../../DATASETS/NA61_EPOS/all/nrg24.npy')
print(nrg_ep.shape)
sh_size = nrg_sh.size
ep_size = nrg_ep.size

nrg_sh_sort = np.sort(nrg_sh)
nrg_ep_sort = np.sort(nrg_ep)

perc_sh = 23.9
perc_ep = 24.9

sh_bound = nrg_sh_sort[int(sh_size * perc_sh / 100)]
ep_bound = nrg_ep_sort[int(ep_size * perc_ep / 100)]
print(sh_bound)
print(ep_bound)

sh_ep_lab_flat = np.zeros((sh_size))
sh_ep_lab_flat[nrg_sh > ep_bound] = 1
ep_sh_lab_flat = np.zeros((ep_size))
ep_sh_lab_flat[nrg_ep > sh_bound] = 1

sh_ep_lab = np.zeros((sh_size, 2))
sh_ep_lab[sh_ep_lab_flat == 0, 0] = 1
sh_ep_lab[sh_ep_lab_flat == 1, 1] = 1

ep_sh_lab = np.zeros((ep_size, 2))
ep_sh_lab[ep_sh_lab_flat == 0, 0] = 1
ep_sh_lab[ep_sh_lab_flat == 1, 1] = 1

#np.save('../../DATASETS/NA61_SHIELD/all/labels_nrg_EPOS.npy', sh_ep_lab)
#np.save('../../DATASETS/NA61_EPOS/all/labels_nrg_SHIELD.npy', ep_sh_lab)