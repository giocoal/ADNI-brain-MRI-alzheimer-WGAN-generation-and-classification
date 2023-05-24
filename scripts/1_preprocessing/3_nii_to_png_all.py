# Converts the volumetric data (.nii) to .png files.
# Creates a list of all the file names, and for each instance:
# 	Because of the complex structure of the original files, it creates a directory on the corresponding {ad, nor} png root directory
# 	Converts the single volumetric file of a visit to multiple .png files
# 	Keeps record of the volumetric dimensions to a .csv file

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image as Image
import pandas as pd
from tqdm import tqdm

import os
from os import walk, path, makedirs

# dtype = "ad" # "ad" (AD patient) or "nor" (Normal)
# img_dir   = "ttv_split_{}_png".format(dtype)
# nifti_dir = "ttv_split_{}/".format(dtype)


# a random sample name is "ttv_split_ad/test/ad/005_S_0814/MPR__GradWarp__B1_Correction__N3__Scaled/2006-08-30_09_32_32.0/S18390/...
# .../ADNI_005_S_0814_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070923123111793_S18390_I74591.nii"

def create_directory(filename, img_dir):
	dir_list = filename.rsplit("\\", 1)

	name = dir_list[-1]
	name = name.rsplit('.', 1)[0] # get name of visit file, ignoring the .nii

	dir_list = dir_list[:-1] # in order to recreate the original directory structure of the dataset
	
	current_dir = img_dir
	for next_dir in dir_list: # iteratively build identical directory structure on the png root directory
		current_dir = current_dir + '/' + next_dir
		if not path.isdir(current_dir):
			makedirs(current_dir)

	for_csv = current_dir
	return for_csv, current_dir + '/' + name

def make_images(destination, data):
	images = []
	array  = []
	for i in range(data.shape[0]):
		array = np.array(data[i,:,:].tolist()) # get an axial slice from the volumetric data, as a numpy array
		images.append(array) # the subject's head is a list of axial slices

	norm_images = []
	for array in images: # normalize the images
		max_element = np.amax(array)
		if max_element > 0:
			array = (array/max_element) * 255.0
		norm_images.append(array)

	images = norm_images
	for img, i in zip(images, range(len(images))):
		#print(img.shape, i)
		im = Image.fromarray(img)
		if (im.mode != 'L'):
			im = im.convert('L') # image is black and white
		im.save(destination + '_{}.png'.format(len(images) -1 -i)) # .nii data start from top to bottom, we wanted bottom to top

# Create the main structure
if __name__ == '__main__':
    
	os.chdir("./data/adni-images")
	dtypes = ["ad", "nor", "mci"] # "ad" (AD patient) or "nor" (Normal)

	for dtype in dtypes:
		
		print("Processing {} data".format(dtype))
		img_dir = "ttv_split_{}_png".format(dtype)
		#img_dir = path.join(data_path, img_dir)
		nifti_dir = "ttv_split_{}/".format(dtype)
		#nifti_path = path.join(data_path, nifti_dir)

		filenames = []

		for root, _, files in walk(nifti_dir):
			for file in files:
				if file.endswith('.nii'):
					filenames.append(path.join(root, file))


		name_list  = []
		count_list = []
  
		print("Total number of files: ", len(filenames))

		for name in tqdm(filenames):
		#	print(name)
			img  = nib.load(name) # load image
			data = np.asanyarray(img.dataobj) # convert it to a numpy array
		#	data_list.append(data)
			for_csv, destination = create_directory(name, img_dir)
			make_images(destination, data)
			name_list.append(for_csv)
			count_list.append(data.shape)
		#	create_dir(name)

		for name, count in zip(name_list, count_list):
			print(name, count)

		df = pd.DataFrame(data = {"dir_name":name_list, "file_count":count_list})
		df.to_csv("counter_all_ttv_{}.csv".format(dtype), sep = ',', index = False)