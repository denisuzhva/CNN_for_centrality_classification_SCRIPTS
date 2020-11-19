#####################
# Scan .txt dataset #
# for particle ids  #
#####################



import numpy as np



events_file = open('../../DATASETS/txtData/EPOS/BeBe150EPOS_SimPSD.txt', 'r')
ef_strings = events_file.readlines()
events_file.close()
last_string_index = len(ef_strings)


def eventSepFinder():
    '''
        Search for event separator
    '''
    srch_str = '*'
    event_separators = np.array([])
    ef_st_iter = 0
    for ef_st in ef_strings:
        if srch_str in ef_st:
            event_separators = np.append(
                event_separators, int(ef_st_iter))
            #print(ef_st_iter)
        ef_st_iter = ef_st_iter + 1
    event_separators = np.append(
        event_separators, last_string_index)  # cuz no * in the end
    return event_separators


def labDatSepFinder():
    '''
        Search for label-data separator
    '''
    srch_str = 'module 1\t'
    labdat_separators = np.array([])
    ef_st_iter = 0
    for ef_st in ef_strings:
        if srch_str in ef_st:
            labdat_separators = np.append(
                labdat_separators, int(ef_st_iter))
            #print(ef_st_iter-1)
        ef_st_iter = ef_st_iter + 1
    return labdat_separators


class Event:
    '''
        Class Of a Single Event
    '''
    def __init__(self, label_start, data_start, data_end):
        self.l_start = label_start
        self.d_start = data_start
        self.d_end = data_end
        #self.f_strings = f_strings

    def findTrueLabel(self):
        srch_start = self.l_start + 1
        srch_end = self.d_start - 2
        #srch_range = list(range(srch_start, srch_end))
        #srch_area_of_strings = np.take(ef_strings, srch_range)
        srch_area_of_strings = ef_strings[srch_start:srch_end]
        #print(srch_area_of_strings)
        return srch_area_of_strings

        # !!! then we need to somehow process the values in order to get the label-centrality

    #def findDataTensor(self):


'''
ev_sep_arr = eventSepFinder()
ld_sep_arr = labDatSepFinder()
#print(ev_sep_arr)
ev_sep = int(ev_sep_arr[0])
ld_sep = int(ld_sep_arr[0])
ld_end = int(ev_sep_arr[1])

#print(len(ev_sep_arr))


a_single_event = Event(ev_sep, ld_sep, ld_end)
event_particles = a_single_event.findTrueLabel()
print(event_particles[0])
particle_iter = 0
for ev_p_string in event_particles:

    particle_energy = ev_p_string.split('	')[1]
    print(particle_energy)

    particle_id = ev_p_string.split('	')[3]
    print(particle_id)
'''


'''
    !!!HERE WE GO!!!
'''

ev_sep_arr = eventSepFinder()
ld_sep_arr = labDatSepFinder()
n_of_events = len(ld_sep_arr)
noe_range = list(range(n_of_events))
print(n_of_events)

list_of_particle_ids = []


for i in range(n_of_events):

    ev_sep = int(ev_sep_arr[i])
    ld_sep = int(ld_sep_arr[i])
    ld_end = int(ev_sep_arr[i+1])

    single_event = Event(ev_sep, ld_sep, ld_end)
    event_particles = single_event.findTrueLabel()
    nucleon_iter = 0
    for ev_p_string in event_particles:
        particle_id = ev_p_string.split('	')[3]
        #print(ev_p_string.split('	'))
        particle_id = particle_id[0:len(particle_id)]
        if particle_id not in list_of_particle_ids:
            list_of_particle_ids.append(particle_id)


print('Id list: ', list_of_particle_ids)

