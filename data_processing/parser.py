####################################
##### WARNING ##### DEPRECATED #####
####################################

# This script creates datasets as #
# .npy tensors out of raw .txt    #
# data made of CERN .root files   #
###################################



import numpy as np
import time


def eventSepFinder():
    """Search for event separator."""
    srch_str = "*"
    event_separators = np.array([])
    ef_st_iter = 0
    for ef_st in ef_strings:
        if srch_str in ef_st:
            event_separators = np.append(
                event_separators, int(ef_st_iter))
        ef_st_iter = ef_st_iter + 1
    event_separators = np.append(
        event_separators, last_string_index)  # cuz no * in the end
    return event_separators


def lafDatSepFinder():
    """Search for label-feature separator."""
    srch_str = "module 1\t"
    lafdat_separators = np.array([])
    ef_st_iter = 0
    for ef_st in ef_strings:
        if srch_str in ef_st:
            lafdat_separators = np.append(
                lafdat_separators, int(ef_st_iter))
        ef_st_iter = ef_st_iter + 1
    return lafdat_separators


def remap3D(ev_mat, in_depth):
    """Map of NA61 PSD modules to 3D np.array."""
    ev_tensor = np.zeros((12, 12, in_depth), dtype=np.float32)
    mat_range = range(28, 32)
    for mat_iter in mat_range:
        doubled_iter = mat_iter*2
        ev_tensor[0][doubled_iter + 2 - 56] = ev_mat[mat_iter]
        ev_tensor[0][doubled_iter + 3 - 56] = ev_mat[mat_iter]
        ev_tensor[1][doubled_iter + 2 - 56] = ev_mat[mat_iter]
        ev_tensor[1][doubled_iter + 3 - 56] = ev_mat[mat_iter]
    mat_range = range(36, 40)
    for mat_iter in mat_range:
        doubled_iter = mat_iter*2
        if mat_iter == 36:
            psd_iter = 39
        if mat_iter == 37:
            psd_iter = 38
        if mat_iter == 38:
            psd_iter = 37
        if mat_iter == 39:
            psd_iter = 36
        ev_tensor[10][doubled_iter + 2 - 72] = ev_mat[psd_iter]
        ev_tensor[10][doubled_iter + 3 - 72] = ev_mat[psd_iter]
        ev_tensor[11][doubled_iter + 2 - 72] = ev_mat[psd_iter]
        ev_tensor[11][doubled_iter + 3 - 72] = ev_mat[psd_iter]
    mat_range = range(32, 36)
    for mat_iter in mat_range:
        doubled_iter = mat_iter*2
        ev_tensor[doubled_iter + 2 - 64][10] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 64][10] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 2 - 64][11] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 64][11] = ev_mat[mat_iter]
    mat_range = range(40, 44)
    for mat_iter in mat_range:
        doubled_iter = mat_iter*2
        if mat_iter == 40:
            psd_iter = 43
        if mat_iter == 41:
            psd_iter = 42
        if mat_iter == 42:
            psd_iter = 41
        if mat_iter == 43:
            psd_iter = 40
        ev_tensor[doubled_iter + 2 - 80][0] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 80][0] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 2 - 80][1] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 80][1] = ev_mat[psd_iter]
    mat_range = range(16, 19)
    for mat_iter in mat_range:
        doubled_iter = mat_iter*2
        ev_tensor[2][doubled_iter + 2 - 32] = ev_mat[mat_iter]
        ev_tensor[2][doubled_iter + 3 - 32] = ev_mat[mat_iter]
        ev_tensor[3][doubled_iter + 2 - 32] = ev_mat[mat_iter]
        ev_tensor[3][doubled_iter + 3 - 32] = ev_mat[mat_iter]
    mat_range = range(19, 22)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[doubled_iter + 2 - 38][8] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 38][8] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 2 - 38][9] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 38][9] = ev_mat[mat_iter]
    mat_range = range(22, 25)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 22:
            psd_iter = 24
        if mat_iter == 23:
            psd_iter = 23
        if mat_iter == 24:
            psd_iter = 22
        ev_tensor[8][doubled_iter + 2 - 44] = ev_mat[psd_iter]
        ev_tensor[8][doubled_iter + 3 - 44] = ev_mat[psd_iter]
        ev_tensor[9][doubled_iter + 2 - 44] = ev_mat[psd_iter]
        ev_tensor[9][doubled_iter + 3 - 44] = ev_mat[psd_iter]
    mat_range = range(25, 28)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 25:
            psd_iter = 27
        if mat_iter == 26:
            psd_iter = 26
        if mat_iter == 27:
            psd_iter = 25
        ev_tensor[doubled_iter + 2 - 50][2] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 50][2] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 2 - 50][3] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 50][3] = ev_mat[psd_iter]
    mat_range = range(0, 4)
    for mat_i in mat_range:
        for mat_j in mat_range:
            ev_tensor[4 + mat_i][4 + mat_j] = ev_mat[mat_j + mat_i * 4]
    return ev_tensor


def remap3D8x8(ev_mat, in_depth):
    """Map of NA61 PSD modules to 3D np.array."""
    ev_tensor = np.zeros((8, 8, in_depth), dtype=np.float32)
    mat_range = range(17, 19)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[0][doubled_iter + 2 - 34] = ev_mat[mat_iter]
        ev_tensor[0][doubled_iter + 3 - 34] = ev_mat[mat_iter]
        ev_tensor[1][doubled_iter + 2 - 34] = ev_mat[mat_iter]
        ev_tensor[1][doubled_iter + 3 - 34] = ev_mat[mat_iter]
    mat_range = range(20, 22)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[doubled_iter + 2 - 40][6] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 40][6] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 2 - 40][7] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 40][7] = ev_mat[mat_iter]
    mat_range = range(23, 25)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 23:
            psd_iter = 24
        if mat_iter == 24:
            psd_iter = 23
        ev_tensor[6][doubled_iter + 2 - 46] = ev_mat[psd_iter]
        ev_tensor[6][doubled_iter + 3 - 46] = ev_mat[psd_iter]
        ev_tensor[7][doubled_iter + 2 - 46] = ev_mat[psd_iter]
        ev_tensor[7][doubled_iter + 3 - 46] = ev_mat[psd_iter]
    mat_range = range(26, 28)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 26:
            psd_iter = 27
        if mat_iter == 27:
            psd_iter = 26
        ev_tensor[doubled_iter + 2 - 52][0] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 52][0] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 2 - 52][1] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 52][1] = ev_mat[psd_iter]
    mat_range = range(0, 4)
    for mat_i in mat_range:
        for mat_j in mat_range:
            ev_tensor[2 + mat_i][2 + mat_j] = ev_mat[mat_j + mat_i * 4]
    return ev_tensor


def remap3D8x8_mini(ev_mat, in_depth):
    """Map of NA61 PSD modules to 3D np.array."""
    ev_tensor = np.zeros((8, 8, in_depth), dtype=np.float32)
    mat_range = range(17, 19)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[1][doubled_iter + 2 - 34] = ev_mat[mat_iter]
        ev_tensor[1][doubled_iter + 3 - 34] = ev_mat[mat_iter]
    mat_range = range(20, 22)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        ev_tensor[doubled_iter + 2 - 40][6] = ev_mat[mat_iter]
        ev_tensor[doubled_iter + 3 - 40][6] = ev_mat[mat_iter]
    mat_range = range(23, 25)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 23:
            psd_iter = 24
        if mat_iter == 24:
            psd_iter = 23
        ev_tensor[6][doubled_iter + 2 - 46] = ev_mat[psd_iter]
        ev_tensor[6][doubled_iter + 3 - 46] = ev_mat[psd_iter]
    mat_range = range(26, 28)
    for mat_iter in mat_range:
        doubled_iter = mat_iter * 2
        if mat_iter == 26:
            psd_iter = 27
        if mat_iter == 27:
            psd_iter = 26
        ev_tensor[doubled_iter + 2 - 52][1] = ev_mat[psd_iter]
        ev_tensor[doubled_iter + 3 - 52][1] = ev_mat[psd_iter]
    mat_range = range(0, 4)
    for mat_i in mat_range:
        for mat_j in mat_range:
            ev_tensor[2 + mat_i][2 + mat_j] = ev_mat[mat_j + mat_i * 4]
    return ev_tensor



class Event:
    """Class Of a Single Event."""
    def __init__(self, in_psd_depth, label_start, mul_str, data_start, data_end):
        self._l_start = label_start
        self._m_str = mul_str
        self._d_start = data_start
        self._d_end = data_end
        self._p_depth = in_psd_depth

    def findSpecLabel(self):
        srch_start = self._l_start + 1
        srch_end = self._d_start - 2
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        spect_number = 0
        for ev_p_string in srch_area_of_strings:
            particle_id = ev_p_string.split('	')[3]
            particle_id = particle_id[0:len(particle_id)]
            particle_id = int(particle_id)
            particle_energy = ev_p_string.split('	')[1]
            particle_energy = float(particle_energy)
            #particle_position = ev_p_string.split('	')[4]
            #particle_position = float(particle_position[0:len(particle_position)-1])
            particle_position = 100500.0

            if particle_position == 100500.0:
                if (particle_id == 2212) or (particle_id == 2112):
                    if particle_energy > 120:
                        spect_number = spect_number + 1
                else:
                    spect_number = spect_number + dict_of_particles.get(particle_id)
        if spect_number <= 3:
            event_class = 0
        else:
            event_class = 1
        return event_class

    def findNrgLabel(self):
        srch_start = self._l_start + 1
        srch_end = self._d_start - 2
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        nrg = 0
        for ev_p_string in srch_area_of_strings:
            particle_energy = ev_p_string.split('	')[1]
            nrg += float(particle_energy)

        if nrg <= 719.67:   # 719.67 to 15.8% centrality
            event_class = 0
        else:
            event_class = 1
        return event_class

    def findMul(self):
        string_num = self._m_str
        srch_srt = ef_strings[string_num]
        print(srch_srt)
        mul_n = srch_srt.split(' ')[-1]
        mul_n = int(mul_n)
        return mul_n


    """
    def findNrgLabel(self):
        srch_start = self._l_start + 1
        srch_end = self._d_start - 1
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        nrg = 0
        for ev_p_string in srch_area_of_strings:
            particle_energy = ev_p_string.split('	')[1]
            particle_position = ev_p_string.split('	')[4]
            particle_position = float(particle_position[0:len(particle_position) - 1])
            nrg += float(particle_energy)

            if particle_position == 100500.0:
                nrg += float(particle_energy)
        if nrg <= 719.67:   # 719.67 to 15.8% centrality
            event_class = 0
        else:
            event_class = 1
        return event_class
    """

    def findDataTensor(self):
        srch_start = self._d_start
        srch_end = self._d_end
        depth = self._p_depth
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((44, depth), dtype=np.float32)

        dps_iter = 0
        if not len(srch_area_of_strings) == 44:
            print(len(srch_area_of_strings))
        for data_p_string in srch_area_of_strings:
            mat_string = data_p_string.split()
            del mat_string[0:2]
            mat_string = [float(ms_iter) for ms_iter in mat_string]
            event_mat[dps_iter] = np.asarray(mat_string)
            dps_iter = dps_iter + 1

        event_tens = remap3D8x8(event_mat, depth)   # choose wisely
        return event_tens

    def findDataMat(self):
        srch_start = self._d_start
        srch_end = self._d_end
        depth = self._p_depth
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((44, depth), dtype=np.float32)

        dps_iter = 0
        if not len(srch_area_of_strings) == 44:
            print(len(srch_area_of_strings))
        for data_p_string in srch_area_of_strings:
            mat_string = data_p_string.split()
            del mat_string[0:2]
            mat_string = [float(ms_iter) for ms_iter in mat_string]
            event_mat[dps_iter] = np.asarray(mat_string)
            dps_iter = dps_iter + 1

        return event_mat



"""!!!HERE WE GO!!!"""

if __name__ == '__main__':

    start_time = time.time()

    dict_of_particles = {2212:1, 2112:1, -2212:0, -2112:0,
                     1000010020:2, 1000010030:3, 1000020030:3,
                     1000020040:4, 1000020060:6, 1000030060:6,
                     1000030070:7, 1000030080:8, 1000040070:7,
                     211:0, -211:0, 111:0, 321:0, -321:0,
                     311:0, -311:0, 130:0, 310:0,
                     3112:0, -3112:0, 3122:0, 3312:0, -3312:0, 3322:0, -3322:0, 3334:0,
                     11:0, -11:0, 22:0}



    #events_file_path = "./txtData/2018_10_15_MLPSDHandler_LiBe_150_ready.txt" # old dataset
    #events_file_path = "./txtData/2018_11_27_MLPSDHandler_LiBe_150_ready.txt"
    events_file_path = "../../DATASETS/txtData/2018_10_15_MLPSDHandler_LiBe_150.txt"
    #events_file_path = "./txtData/t3e.txt"

    dataset_size = "full"   # DATASET_TYPE

    events_file = open(events_file_path, "r")
    ef_strings = events_file.readlines()
    events_file.close()
    last_string_index = len(ef_strings)

    psd_depth = 20
    dat_dim = 8

    ev_sep_arr = eventSepFinder().astype(int)
    ld_sep_arr = lafDatSepFinder().astype(int)
    mul_sep_arr = lafDatSepFinder().astype(int) - 1

    n_of_events = None
    train_size = None
    test_size = None
    if dataset_size == 'full':
        n_of_events = len(ld_sep_arr)
        train_size = 80000
        test_size = 19500
    if dataset_size == 'mid':
        n_of_events = 9000
        train_size = 5000
        test_size = 4000

    assert train_size + test_size == n_of_events

    noe_range = list(range(n_of_events))
    train_dataset = np.zeros((train_size, dat_dim*dat_dim*psd_depth), dtype=np.float32)
    train_labels_spec = np.zeros([train_size, 2], dtype=np.float32)
    train_labels_nrg = np.zeros([train_size, 2], dtype=np.float32)
    test_dataset = np.zeros((test_size, dat_dim*dat_dim*psd_depth), dtype=np.float32)
    test_labels_spec = np.zeros([test_size, 2], dtype=np.float32)
    test_labels_nrg = np.zeros([test_size, 2], dtype=np.float32)
    all_mul = np.zeros(n_of_events, dtype=np.int16)

    for i in range(n_of_events):
        ev_sep = ev_sep_arr[i]
        mul_sep = mul_sep_arr[i]
        lf_sep = ld_sep_arr[i]
        lf_end = ev_sep_arr[i+1]
        single_event = Event(psd_depth, ev_sep, mul_sep, lf_sep, lf_end)
        #all_mul[i] = single_event.findMul()
        if i < train_size:
            train_labels_spec[i][single_event.findSpecLabel()] = 1
            train_labels_nrg[i][single_event.findNrgLabel()] = 2
            train_dataset[i] = single_event.findDataTensor().flatten()
        else:
            test_labels_spec[i - train_size - 1][single_event.findSpecLabel()] = 1
            test_labels_nrg[i - train_size - 1][single_event.findNrgLabel()] = 1
            test_dataset[i - train_size - 1] = single_event.findDataTensor().flatten()
        del single_event

    """
    max_hit_train = np.max(train_dataset)
    max_hit_test = np.max(test_dataset)
    if max_hit_train > max_hit_test:
    train_dataset = np.true_divide(train_dataset, max_hit_train)
    test_dataset = np.true_divide(test_dataset, max_hit_train)
    else:
    train_dataset = np.true_divide(train_dataset, max_hit_test)
    test_dataset = np.true_divide(test_dataset, max_hit_test)
    """

    print(train_dataset.shape)
    print(train_labels_spec.shape)
    print(train_labels_nrg.shape)
    print(test_dataset.shape)
    print(test_labels_spec.shape)
    print(test_labels_nrg.shape)
    #np.save('../AI/NA61_minicross10_8x8/{0}/train_dataset_flat.npy'.format(dataset_size), train_dataset)
    #np.save('../AI/NA61_minicross10_8x8/{0}/train_labels_spec_flat.npy'.format(dataset_size), train_labels_spec)
    #np.save('../AI/NA61_minicross10_8x8/{0}/train_labels_nrg_flat.npy'.format(dataset_size), train_labels_nrg)

    #np.save('../AI/NA61_minicross10_8x8/{0}/test_dataset_flat.npy'.format(dataset_size), test_dataset)
    #np.save('../AI/NA61_minicross10_8x8/{0}/test_labels_spec_flat.npy'.format(dataset_size), test_labels_spec)
    #np.save('../AI/NA61_minicross10_8x8/{0}/test_labels_nrg_flat.npy'.format(dataset_size), test_labels_nrg)

    #np.save('../AI/NA61_minicross10_8x8/{0}/all_mul.npy'.format(dataset_size), all_mul)


    print("--- %s seconds ---" % (time.time() - start_time))

