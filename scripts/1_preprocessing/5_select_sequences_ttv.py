# The effects of AD are visible mostly on the middle part of the head, with the lower and upper portions being irrelevant.
# With that in mind, we empirically chose some thresholds and only kept the relevant middle parts (from the ~45 percentile slice to the ~80 percentile slice).
# However, the original volumetric data, and thus the .png data we used, did not have the same dimensions
# i.e. the subjects' heads were not split in the same number of slices, and said slices did not have the same dimensions.
# So, if the visit had 192 slices, only slices 92-140 were kept, etc.

import shutil
import os 
from os import walk, path
from  tqdm import tqdm

if __name__ == '__main__':
	os.chdir("./data/adni-images")
	dtypes = ["nor", "ad", "mci"]
	for dtype in dtypes:
		print("Processing {} data".format(dtype))
		png_dir = "ttv_split_{}_png/".format(dtype)
		seq_dir = "ttv_split_{}_png_seq/".format(dtype)
		for root, _, files in walk(png_dir):
		#	[print(file) if (str(file).endswith("{}.png".format(str(i))) for i in range(50,150)) for file in files]
			for file in files:
				if "test" in root:
					split = "test/{}/".format(dtype)
				elif "train" in root:
					split = "train/{}/".format(dtype)
				elif "valid" in root:
					split = "valid/{}/".format(dtype)

				if not os.path.exists(path.join(seq_dir, split)):
					os.makedirs(path.join(seq_dir, split))

				file_lst = file.rsplit("_", 1) # get {number}.png
				sequence = file_lst[-1]
				sequence = sequence.rsplit('.', 1)[0] # get {number}

				if len(files) == 192:
					if int(sequence) >= 95 and int(sequence) <= 140:
						shutil.copy(path.join(root, file), path.join(seq_dir, split))
				elif len(files) == 240:
					if int(sequence) >= 100 and int(sequence) <= 180:
						shutil.copy(path.join(root, file), path.join(seq_dir, split))
				elif len(files) == 256:
					if int(sequence) >= 120 and int(sequence) <= 200:
						shutil.copy(path.join(root, file), path.join(seq_dir, split))
