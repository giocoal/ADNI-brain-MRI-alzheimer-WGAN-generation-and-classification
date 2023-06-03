"""
Randomly splits subjects on training, validation and test sets. 
If it is specified, it also performs random undersampling to balance the classes.
"""


import numpy as np

from os import listdir, path
from shutil import copytree

sampling_method = "full" # ["full", "random undersampling"]

nii_dir = "nor" # ["mci", "ad", "nor"] original directories were "ad" (AD patient), "nor" (Normal) and mci (mild cognitive impairment)
split_dir = "ttv_split_{}".format(nii_dir)

nii_path = path.join("./data/adni-mri", nii_dir)
split_dir = path.join("./data/adni-images", split_dir)

train_dir = path.join(split_dir, "train")
test_dir  = path.join(split_dir, "test")
valid_dir = path.join(split_dir, "valid")

patients = listdir(nii_path)

for patient in patients:
	if "nor" in nii_dir:
		if sampling_method == "full":
			keep_patient = True
		elif sampling_method == "random undersampling":
			keep_patient = np.random.rand() <= 0.8492 # to balance Normal subjects with AD patients
	elif "mci" in nii_dir:
		if sampling_method == "full":
			keep_patient = True 
		elif sampling_method == "random undersampling":
			keep_patient = np.random.rand() <= 0.4975 # to balance MCI subjects with AD patients
	else:
		keep_patient = True
  
	if keep_patient:
		split = np.random.rand()

		if split <= 0.8333: # ~25.000/30.000 images to train set
			current_dir = train_dir
			src = path.join(nii_path, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("train")
		elif split <=0.9333: # ~3.000/30.000 images to valid set
			current_dir = valid_dir
			src = path.join(nii_path, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("valid")
		else: # ~2.000/30.000 images to test set
			current_dir = test_dir
			src = path.join(nii_path, patient)
			dst = path.join(current_dir, path.join(nii_dir, patient))

			copytree(src,dst)
#			print("test")
