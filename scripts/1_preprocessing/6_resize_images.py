# Resize all images underneath directory 'data' to (160, 192), through Lanczos resampling.

import numpy as np

from os import path, walk
from PIL import Image
import os

if __name__ == '__main__':
	os.chdir("./data/adni-images")
	dtypes = ["nor", "ad", "mci"]
	for dtype in dtypes:
		print("Processing {} data".format(dtype))
		seq_dir = "ttv_split_{}_png_seq/".format(dtype)
		for root, _, files in walk(seq_dir):
			for file in files:
				img = Image.open(path.join(root, file))
				img = img.resize((160,192), resample = Image.LANCZOS)
				dst = root.replace("_seq", "_seq_resized")
				if not os.path.exists(dst):
					os.makedirs(dst)
				img.save(path.join(dst, file))
