import numpy as np



cen = np.load('../../DATASETS/NA61_EPOS/all/features_central.npy')
per = np.load('../../DATASETS/NA61_EPOS/all/features_peripheral.npy')
ev_mat = np.load('../../DATASETS/NA61_EPOS/all/ev_mat.npy')

trs = 0.5
cross_zeros = np.sum(cen < trs*np.mean(cen)) + np.sum(per < trs*np.mean(per))
cross_features = cen.size + per.size
cross_sp_c = cross_zeros / cross_features
print(cross_sp_c)

full_zeros = np.sum(ev_mat < trs*np.mean(ev_mat))
full_features = ev_mat.size
full_sp_c = full_zeros / full_features
print(full_sp_c)

#cross_zeros = np.sum(cen == 0) + np.sum(per == 0)
#cross_features = cen.size + per.size
#
#print(cross_zeros)
#print(cross_features)
#
#cross_sp_c = cross_zeros / cross_features
#print(cross_sp_c)
#
#full_zeros = np.sum(ev_mat == 0)
#full_features = ev_mat.size
#
#full_sp_c = full_zeros / full_features
#print(full_sp_c)