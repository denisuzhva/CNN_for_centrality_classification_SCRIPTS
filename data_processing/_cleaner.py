# simple cleaner (removes "/*" artifacts)

events_file_path = "./txtData/2018_11_27_MLPSDHandler_LiBe_150.txt"
events_file_path_cleaned = "./txtData/2018_11_27_MLPSDHandler_LiBe_150_ready.txt"

events_file = open(events_file_path, 'r')
ef_strings = events_file.readlines()
events_file.close()

events_file = open(events_file_path_cleaned, 'w')
for line in ef_strings:
    if not line[0] == '/':
        events_file.write(line)
events_file.close()
