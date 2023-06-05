# A parser, for train_resnet.py

import argparse
import os
import keras.backend as K

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data_gan_adnorfirst', help='directory where data are located')
parser.add_argument('--target_size', type=int, default=(192, 160), help='image dimensions')
parser.add_argument('--weights_dir', type=str, default=None, help='where to store weights')
parser.add_argument('--log_dir', type=str, default='logs', help='where to store tensorboard logs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--aug_prob', type=float, default=0.0, help='augmentation probability of iaa.Sometimes')
parser.add_argument('--dropout_prob', type=float, default=0.25, help='dropout probability')
parser.add_argument('--fake_ratio', type=float, default=0.5, help='ratio of fake and real images')
parser.add_argument('--neurons', type=int, default=0, help='number of fully_connected_neurons before softmax')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--mem_perc', type=float, default=0.2, help='percentage of memory to be reserved for the model')
parser.add_argument('--backend_format', type=str, default='channels_last', help='keras image data format')
#parser.add_argument('--num_classes', type=int, default=2, help='number of classes in the dataset')
#parser.add_argument('--pretrained_weights', type=str, default=None, help='path to a valid weight file')
#parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


K.set_image_data_format(opt.backend_format)