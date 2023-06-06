# It can be done in GIMP, but I am a hackerman

import numpy as np

from os import listdir
from PIL import Image
from random import shuffle

dtype = 'nor'
whitespace = 1
side = 6
img_loc = 'some_{}/'.format(dtype)

#img_list = [7068, 8010, 8955, 10067, 12786, 14491]
img_list = listdir(img_loc)
shuffle(img_list)

#name_list = [(img_loc + '_{}.png'.format(i)) for i in img_list]
name_list = img_list[:side*side]
print(name_list)


stitch = np.zeros(shape = (192*side + whitespace*(side - 1), 160*side + whitespace*(side - 1)), dtype = np.uint8)

white_col = np.full(shape = (192,whitespace), fill_value = 255, dtype = np.uint8)
white_row = np.full(shape = (whitespace, 160*side + whitespace*(side - 1)), fill_value = 255, dtype = np.uint8)

for row in range(side):
	for col in range(side): # horizontally, then vertically
		#fill columns
		img = Image.open(img_loc + name_list[col*side + row])
		arr = np.asarray(img)

		stitch[row*(192 + whitespace):(row+1)*192 + row*whitespace,col*(160 + whitespace):(col+1)*160 + col*whitespace] = arr
		
		if col < (side - 1):
			stitch[row*192 + row*whitespace:(row+1)*192 + row*whitespace,(col+1)*160 + col*whitespace:(col+1)*160 + (col+1)*whitespace] = white_col

	if row < (side - 1):
		stitch[(row + 1)*192 + row*whitespace:(row + 1)*192 + (row + 1)*whitespace,:] = white_row


stitch_img = Image.fromarray(stitch)
stitch_img.save('gan_images_{}.png'.format(dtype))