import numpy as np
import time

start_time = time.time()


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


def remap3D(ev_mat):
    """Map of NA61 PSD modules to 3D np.array."""
    ev_tensor = np.zeros((12, 12, 20), dtype=np.float16)
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


class Event:
    """Class Of a Single Event."""
    def __init__(self, label_start, data_start, data_end):
        self.l_start = label_start
        self.d_start = data_start
        self.d_end = data_end

    def findDataMat(self, list_fine):
        srch_start = self.d_start
        srch_end = self.d_end
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((24, 20), dtype=np.float16)

        dps_iter = 0
        if not len(srch_area_of_strings) == 44:
            print(len(srch_area_of_strings))
        for data_p_string in srch_area_of_strings:
            mat_string = data_p_string.split()
            module_num = int(mat_string[1])
            if module_num in list_fine:
                del mat_string[0:2]
                mat_string = [float(ms_iter) for ms_iter in mat_string]
                event_mat[dps_iter] = np.asarray(mat_string)
                dps_iter = dps_iter + 1

        return event_mat


"""!!!HERE WE GO!!!"""


dict_of_particles = {"2212":1, "2112":1, "-2212":0, "-2112":0,
                     "1000010020":2, "1000010030":3, "1000020030":3,
                     "1000020040":4, "1000020060":6, "1000030060":6,
                     "1000030070":7, "1000030080":8, "1000040070":7,
                     "211":0, "-211":0, "111":0, "321":0, "-321":0,
                     "311":0, "-311":0, "22":0}
list_of_accepted_modules = list(range(16))
list_of_accepted_modules = list_of_accepted_modules + [17, 18, 20, 21, 23, 24, 26, 27]

#events_file_path = "./txtData/2018_10_15_MLPSDHandler_LiBe_150_ready.txt" # old dataset
events_file_path = "../dataPrep/txtData/2018_11_27_MLPSDHandler_LiBe_150_ready.txt"
#events_file_path = "./txtData/t3e.txt"
dataset_size = "full"
events_file = open(events_file_path, "r")
ef_strings = events_file.readlines()
events_file.close()
last_string_index = len(ef_strings)

ev_sep_arr = eventSepFinder()
ld_sep_arr = lafDatSepFinder()

n_of_events = len(ld_sep_arr)


noe_range = list(range(n_of_events))
mat_dataset = np.zeros((n_of_events, 24, 20), dtype=np.float32)


for i in range(n_of_events):
    ev_sep = int(ev_sep_arr[i])
    lf_sep = int(ld_sep_arr[i])
    lf_end = int(ev_sep_arr[i+1])
    single_event = Event(ev_sep, lf_sep, lf_end)
    mat_dataset[i] = single_event.findDataMat(list_of_accepted_modules)
    del single_event



np.save('../AI/NA61_better/{0}/mat_dataset.npy'.format(dataset_size), mat_dataset)



print("--- %s seconds ---" % (time.time() - start_time))
