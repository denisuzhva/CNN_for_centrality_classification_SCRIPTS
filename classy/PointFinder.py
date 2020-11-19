#################################
# Build matsums of the datasets #
#################################


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
            event_separators = np.append(event_separators,
                                         int(ef_st_iter))
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


class Event:
    """Class Of a Single Event."""
    def __init__(self, in_psd_depth, label_start,
                 data_start, data_end):
        self.l_start = label_start
        self.d_start = data_start
        self.d_end = data_end
        self.p_depth = in_psd_depth

    def findSpec(self):
        srch_start = self.l_start + 1
        srch_end = self.d_start - 2
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        spect_number = 0
        for ev_p_string in srch_area_of_strings:
            particle_id = ev_p_string.split('	')[3]
            particle_id = particle_id[0:len(particle_id)]
            particle_id = int(particle_id)
            particle_energy = ev_p_string.split('	')[1]
            particle_energy = float(particle_energy)
            particle_position = ev_p_string.split('	')[4]
            particle_position = float(particle_position[0:len(particle_position)-1])

            if particle_position == 100500.0:
                if (particle_id == 2212) or (particle_id == 2112):
                    if particle_energy > 120:
                        spect_number = spect_number + 1
                else:
                    spect_number = spect_number + dict_of_particles.get(particle_id)
        return spect_number

    def findNrg(self):
        srch_start = self.l_start + 1
        srch_end = self.d_start - 2
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        nrg = 0
        for ev_p_string in srch_area_of_strings:
            particle_energy = ev_p_string.split('	')[1]
            nrg += float(particle_energy)
        return nrg

    """
    def findNrgLabel(self):
        srch_start = self.l_start + 1
        srch_end = self.d_start - 1
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

    def findDataTensorSum(self, list_fine):
        srch_start = self.d_start
        srch_end = self.d_end
        depth = self.p_depth
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        event_mat = np.zeros((len(list_fine), depth), dtype=np.float32)

        dps_iter = 0
        if not len(srch_area_of_strings) == 44:
            print(len(srch_area_of_strings))

        for data_p_string in srch_area_of_strings:
            mat_string = data_p_string.split()
            module_num = int(mat_string[1])
            if module_num - 1 in list_fine:
                del mat_string[0:2]
                mat_string = [float(ms_iter) for ms_iter in mat_string]
                event_mat[dps_iter] = np.asarray(mat_string)
                dps_iter = dps_iter + 1
        event_mat_sum = np.sum(event_mat)

        return event_mat_sum


"""!!!HERE WE GO!!!"""


dict_of_particles = {2212:1, 2112:1, -2212:0, -2112:0,
                     1000010020:2, 1000010030:3, 1000020030:3,
                     1000020040:4, 1000020060:6, 1000030060:6,
                     1000030070:7, 1000030080:8, 1000040070:7,
                     211:0, -211:0, 111:0, 321:0, -321:0,
                     311:0, -311:0, 22:0}

list_of_accepted_modules = list(range(16))
list_of_accepted_modules = list_of_accepted_modules + [17, 18, 20, 21, 23, 24, 26, 27]
print("Num of modules: ", len(list_of_accepted_modules))
events_file_path = "../dataPrep/txtData/2019_03_19_MLPSDHandler_LiBe_150_ready.txt"
events_file = open(events_file_path, "r")
ef_strings = events_file.readlines()
events_file.close()
last_string_index = len(ef_strings)

psd_depth = 10

ev_sep_arr = eventSepFinder()
ld_sep_arr = lafDatSepFinder()

n_of_events = len(ld_sep_arr)

mat_sum = np.zeros(n_of_events, dtype=np.float16)
spec_nrg = np.zeros(n_of_events, dtype=np.float16)
spec_num = np.zeros(n_of_events, dtype=np.float16)


for i in range(n_of_events):
    ev_sep = int(ev_sep_arr[i])
    lf_sep = int(ld_sep_arr[i])
    lf_end = int(ev_sep_arr[i+1])
    single_event = Event(psd_depth, ev_sep, lf_sep, lf_end)
    mat_sum[i] = single_event.findDataTensorSum(list_of_accepted_modules)
    spec_num[i] = single_event.findSpec()
    spec_nrg[i] = single_event.findNrg()
    del single_event


if len(list_of_accepted_modules) == 16: # 2019 or 2018... it depends
    np.save('./2019_dataset/mat_sum16.npy', mat_sum)
    np.save('./2019_dataset/spec_num16.npy', spec_num)
    np.save('./2019_dataset/spec_nrg16.npy', spec_nrg)
else:
    np.save('./2019_dataset/mat_sum24.npy', mat_sum)
    np.save('./2019_dataset/spec_num24.npy', spec_num)
    np.save('./2019_dataset/spec_nrg24.npy', spec_nrg)




print("--- %s seconds ---" % (time.time() - start_time))
