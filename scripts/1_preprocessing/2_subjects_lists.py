
"""
Optional script.
This script create, in the subject_list directory, .txt files containing the names of the subjects in the train, test and valid sets 
and for each type of data (AD, MCI and Normal). These files will be used by the data generators to load the data.
"""

import os
from os import listdir

data_path = "./data/adni-images/"
type_dict = {'ad', 'nor', 'mci'}
set_dict = {'train', 'test', 'valid'}
# Create directory to store subjects lists
list_dir = os.path.join(data_path, "subjects_list")
if not os.path.exists(list_dir):
    os.makedirs(list_dir)
print(list_dir)

for dtype in type_dict:
	for dset in set_dict:

		ttv_split_dir = "ttv_split_{}/{}/{}/".format(dtype, dset, dtype)
		name_list = listdir(os.path.join(data_path, ttv_split_dir))
		subjects_path = os.path.join(list_dir, 'subjects_{}_{}.txt'.format(dset, dtype))

		with open(subjects_path, 'w+') as subjects_file:
			
			for name in name_list:
				subjects_file.write(name + "\n")
