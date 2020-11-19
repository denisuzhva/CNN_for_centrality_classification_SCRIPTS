import numpy as np



start_str = 32247702
end_str = 32247704

events_file_path = "../../DATASETS/txtData/realdata/BeBe150_real_data.txt"
events_file_path_clean = "../../DATASETS/txtData/realdata/BeBe150_real_data_clean.txt"

events_file = open(events_file_path, "r")
ef_strings = events_file.readlines()
events_file.close()

print(len(ef_strings))

del ef_strings[start_str:end_str]

print(len(ef_strings))

with open(events_file_path_clean, 'w') as events_file:
    for i in range(len(ef_strings)):
        events_file.write(ef_strings[i])