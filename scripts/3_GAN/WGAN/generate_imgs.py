# Given a .h5 model (not just weights!) as input, produces NUM generated images
import argparse
import os
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
# from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.layers import Convolution2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial


NUM = 33091 # how many images we want to generate
TYPE = 'nor'
batch_size = 256

BATCH_SIZE = 32 # irrelevant for the actual computations, just for naming conventions to access the model
LATENT_SIZE = 128 # latent size used during training, fixed
EPOCHS = 300
MODEL = './scripts/3_GAN/WGAN/weights/generator_{}_newd_{}_{}_{}.h5'.format(TYPE, BATCH_SIZE, LATENT_SIZE, EPOCHS)

ROOT = "./data/generated-images/WGAN/"
IMG_DIR = '{}{}/'.format(ROOT, TYPE) # change at will

print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num CPUs Available: ", len(tf.config.list_logical_devices('GPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# True or False
print(tf.test.is_built_with_gpu_support())
# True or False
print(tf.test.is_built_with_rocm())
# True or False
print(tf.test.is_built_with_xla())
# True or False
print(tf.config.list_physical_devices(device_type='GPU'))
print(tf.config.list_logical_devices(device_type='GPU'))

generator = load_model(MODEL)

n_batch = (NUM // batch_size) + 1 # produce n_batch of images (surplus)
counter_image = 1

for batch_number in range(n_batch):
	print("Generating batch {} of {}".format(batch_number, n_batch))
	noise = np.random.rand(batch_size, LATENT_SIZE)
	gen_imgs = generator.predict(noise)
	gen_imgs = (gen_imgs * 127.5) + 127.5 # rescale
	# lst = []
	# for i in range(NUM):
	# 	lst.append(i)

	images = gen_imgs[:,:,:,0]

	for img in images:
		print("Generating image {} of {}".format(counter_image, NUM))
		#print(img.shape, i)
		im = Image.fromarray(img)
		if (im.mode != 'L'):
			im = im.convert('L')
		im.save(IMG_DIR + 'fake_{}_{}.png'.format(TYPE,counter_image))
		counter_image += 1
