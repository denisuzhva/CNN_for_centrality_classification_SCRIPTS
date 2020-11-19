# This script creates datasets as #
# .npy tensors out of raw .txt    #
# data made of CERN .root files   #
#     **use with real data**      #
###################################



import numpy as np
import time


def eventSepFinder():
    """Search for event separator."""
    srch_str = "\n"
    event_separators = np.array([])
    for ef_st_iter, ef_st in enumerate(ef_strings):
        if srch_str == ef_st:
            event_separators = np.append(
                event_separators, int(ef_st_iter))
            #if event_separators.size > 10:
            #    break
    #event_separators = np.append(event_separators, last_string_index)  # cuz no * in the end
    return event_separators


def remapCenPer(ev_mat, in_depth):
    """Separate the central modules from the peripheral ones."""
    cen_tensor = np.zeros((4, 4, in_depth), dtype=np.float32)
    per_tensor = np.zeros_like(cen_tensor)
    for i in range(4):
        for j in range(4):
            cen_tensor[i, j] = ev_mat[4*i + j]
    per_tensor[0, 1] = ev_mat[17]
    per_tensor[0, 2] = ev_mat[18]
    per_tensor[1, 0] = ev_mat[27]
    per_tensor[2, 0] = ev_mat[26]
    per_tensor[1, 3] = ev_mat[20]
    per_tensor[2, 3] = ev_mat[21]
    per_tensor[3, 1] = ev_mat[24]
    per_tensor[3, 2] = ev_mat[23]
    return cen_tensor, per_tensor


class Event:
    """Class Of a Single Event."""
    def __init__(self, in_psd_depth, mul_str, data_start, data_end):
        self._m_str = mul_str
        self._d_start = data_start
        self._d_end = data_end
        self._p_depth = in_psd_depth

    def findMul(self):
        string_num = self._m_str
        srch_srt = ef_strings[string_num]
        mul_n = srch_srt.split(' ')[-1]
        mul_n = int(mul_n)
        return mul_n

    def findDataTensor(self):
        srch_start = self._d_start
        srch_end = self._d_end
        depth = self._p_depth
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((44, depth), dtype=np.float32)

        dps_iter = 0
        for data_p_string in srch_area_of_strings:
            mat_string = data_p_string.split()
            del mat_string[0:2]
            mat_string = [float(ms_iter) for ms_iter in mat_string]
            if not np.asarray(mat_string).size == depth:
                print(mat_string)
            event_mat[dps_iter] = np.asarray(mat_string)
            dps_iter = dps_iter + 1

        event_tens = remapCenPer(event_mat, depth)   # choose wisely
        return event_tens

    def findDataMat(self):
        srch_start = self._d_start
        srch_end = self._d_end
        depth = self._p_depth
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((44, depth), dtype=np.float32)

        dps_iter = 0
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

    events_file_path = "../../DATASETS/txtData/experData/BeBe150_real_data.txt"

    events_file = open(events_file_path, "r")
    ef_strings = events_file.readlines()
    events_file.close()
    last_string_index = len(ef_strings)

    psd_depth = 10
    dat_dim = 8

    #ev_sep_arr = eventSepFinder().astype(int)
    ev_sep_arr = 46 * np.arange(last_string_index // 46)

    n_of_events = len(ev_sep_arr) - 1
    print(n_of_events)
    n_of_events = 100 * (n_of_events // 100)
    print(n_of_events)

    noe_range = list(range(n_of_events))

    #cen_ft = np.zeros((n_of_events, dat_dim//2, dat_dim//2, psd_depth, 1), dtype=np.float32)
    #per_ft = np.zeros((n_of_events, dat_dim//2, dat_dim//2, psd_depth, 1), dtype=np.float32)
    #sum_ft = np.zeros((n_of_events, 1, 1), dtype=np.float32)
    #all_mul = np.zeros((n_of_events), dtype=np.int32)
    ev_mat = np.zeros((n_of_events, 44, psd_depth), dtype=np.float32)

    for i in range(n_of_events):
        if i % 100000 == 0:
            print(i)
        mul_str = ev_sep_arr[i] + 1
        data_start = mul_str + 1
        data_end = ev_sep_arr[i+1]
        single_event = Event(psd_depth, mul_str, data_start, data_end)
        #all_mul[i] = single_event.findMul()
        #cen_ft[i, :, :, :, 0], per_ft[i, :, :, :, 0] = single_event.findDataTensor()
        ev_mat[i] = single_event.findDataMat()

    #sum_ft = (np.sum(cen_ft, (1, 2, 3, 4)).flatten() + np.sum(per_ft, (1, 2, 3, 4)).flatten()).reshape((-1, 1))

    #print(cen_ft.shape)
    #print(per_ft.shape)
    #print(sum_ft.shape)
    #print(all_mul.shape)
    #np.save('../../DATASETS/NA61_BeBeReal/all/features_central.npy', cen_ft)
    #np.save('../../DATASETS/NA61_BeBeReal/all/features_peripheral.npy', per_ft)
    #np.save('../../DATASETS/NA61_BeBeReal/all/features_sum.npy', sum_ft)
    #np.save('../../DATASETS/NA61_BeBeReal/all/mul.npy', all_mul)
    np.save('../../DATASETS/NA61_BeBeExp/all/ev_mat.npy', ev_mat)

    print("--- %s seconds ---" % (time.time() - start_time))

