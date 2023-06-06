import matplotlib
matplotlib.use('Agg')
# A slightly modified script (sample image generation, loss saving, etc.) of the one found on the Keras-contrib repo
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).

To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.

The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
import argparse
import os
import numpy as np
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial

import matplotlib.pyplot as plt

def average_lst(lst):
    avg_lst = 0
    for x in lst:
        avg_lst += x
    avg_lst /= float(len(lst))
    return avg_lst


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config = config))


TYPE = "ad"
INPUT_LOCATION = "../../gan_array_{}.npy".format(TYPE) # change at will
LATENT_SIZE = 128
EPOCHS = 800
BATCH_SIZE = 32
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
output_dir = "gen_imgs/"
weights_dir = "weights/"
losses_dir = "losses/"


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    model = Sequential()
#    model.add(Dense(512, input_dim=100))
#    model.add(LeakyReLU())
    model.add(Dense(512 * 6 * 5, input_dim = LATENT_SIZE))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((512, 6, 5), input_shape=(512 * 6 * 5,)))
        bn_axis = 1
    else:
        model.add(Reshape((6, 5, 512), input_shape=(512 * 6 * 5,)))
        bn_axis = -1
    model.add(Conv2DTranspose(512, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(256, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(32, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(32, (5, 5), padding='same', input_shape=(1, 192, 160)))
    else:
        model.add(Convolution2D(32, (5, 5), padding='same', input_shape=(192, 160, 1)))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), kernel_initializer='he_normal'))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
class RandomWeightedAverage_fixed(Layer):
    def __init__(self, **kwargs):
        super(RandomWeightedAverage, self).__init__(**kwargs)

    def call(self, inputs):
        weights = tf.random.uniform((tf.shape(inputs[0])[0], 1, 1, 1))
        weighted_inputs = Concatenate(axis=-1)([inputs[0], inputs[1]])
        return tf.reduce_sum(weights * weighted_inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1]


r, c = 5, 5
fixed_noise = np.random.rand(r * c, LATENT_SIZE)
np.save(output_dir + "fixed_{}_{}_{}_{}.npy".format(TYPE, BATCH_SIZE, LATENT_SIZE, EPOCHS), fixed_noise)
# at the end of every epoch, r*c sample images are produced, stacked on a r x c grid
# the images can be produced from a fixed random seed, to monitor improvements on the same input,
# or from a different one every time, for increased variety 
def generate_images(generator_model, output_dir, epoch, fixed = False):
    r, c = 5, 5
    if not fixed:
        noise = np.random.rand(r * c, LATENT_SIZE)
        output_dir += "rand/"
    else:
        noise = fixed_noise
        output_dir += "fixed/"
    gen_imgs = generator_model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = (gen_imgs * 127.5) + 127.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    fig.savefig(outfile)
    plt.close()

# First we load the image data, reshape it and normalize it to the range [-1, 1]
X_train = np.load(INPUT_LOCATION)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
else:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Now we initialize the generator and discriminator.
generator = make_generator()
discriminator = make_discriminator()

generator.summary()
discriminator.summary()
# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(LATENT_SIZE,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(LATENT_SIZE,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

# Create all the log files
d_gen_path = losses_dir + 'disc_gen.txt'
try:
    d_gen_file = open(d_gen_path, 'w')
except IOError:
    # If not exists, create the file
    d_gen_file = open(d_gen_path, 'w+')

d_fake_path = losses_dir + 'disc_fake.txt'
try:
    d_fake_file = open(d_fake_path, 'w')
except IOError:
    # If not exists, create the file
    d_fake_file = open(d_fake_path, 'w+')

d_real_path = losses_dir + 'disc_real.txt'
try:
    d_real_file = open(d_real_path, 'w')
except IOError:
    # If not exists, create the file
    d_real_file = open(d_real_path, 'w+')

d_gradpen_path = losses_dir + 'disc_gradpen.txt'
try:
    d_gradpen_file = open(d_gradpen_path, 'w')
except IOError:
    # If not exists, create the file
    d_gradpen_file = open(d_gradpen_path, 'w+')


d_gen_average_path = losses_dir + 'disc_gen_average.txt'
try:
    d_gen_average_file = open(d_gen_average_path, 'w')
except IOError:
    # If not exists, create the file
    d_gen_average_file = open(d_gen_average_path, 'w+')

d_fake_average_path = losses_dir + 'disc_fake_average.txt'
try:
    d_fake_average_file = open(d_fake_average_path, 'w')
except IOError:
    # If not exists, create the file
    d_fake_average_file = open(d_fake_average_path, 'w+')

d_real_average_path = losses_dir + 'disc_real_average.txt'
try:
    d_real_average_file = open(d_real_average_path, 'w')
except IOError:
    # If not exists, create the file
    d_real_average_file = open(d_real_average_path, 'w+')

d_gradpen_average_path = losses_dir + 'disc_gradpen_average.txt'
try:
    d_gradpen_average_file = open(d_gradpen_average_path, 'w')
except IOError:
    # If not exists, create the file
    d_gradpen_average_file = open(d_gradpen_average_path, 'w+')


g_path = losses_dir + 'gen.txt'
try:
    g_file = open(g_path, 'w')
except IOError:
    # If not exists, create the file
    g_file = open(g_path, 'w+')



for epoch in range(EPOCHS):
    np.random.shuffle(X_train)
#    print("Epoch: ", epoch)
#    print(X_train.shape[0])
#    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        print(epoch, i)
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        d_gen_minibatch_list     = []
        d_fake_minibatch_list    = []
        d_real_minibatch_list    = []
        d_gradpen_minibatch_list = []
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, LATENT_SIZE).astype(np.float32)
            discriminator_minibatch_loss = discriminator_model.train_on_batch([image_batch, noise],
                                                                         [positive_y, negative_y, dummy_y])
            # get training stats
            d_gen_minibatch_loss     = discriminator_minibatch_loss[0]
            d_fake_minibatch_loss    = discriminator_minibatch_loss[1]
            d_real_minibatch_loss    = discriminator_minibatch_loss[2]
            d_gradpen_minibatch_loss = discriminator_minibatch_loss[3]

            # write them on files
            d_gen_file.write(str(d_gen_minibatch_loss) + "\n")
            d_fake_file.write(str(d_fake_minibatch_loss) + "\n")
            d_real_file.write(str(d_real_minibatch_loss) + "\n")
            d_gradpen_file.write(str(d_gradpen_minibatch_loss) + "\n")

            # store them for average calculation after TRAINING_RATIO
            d_gen_minibatch_list.append(d_gen_minibatch_loss)
            d_fake_minibatch_list.append(d_fake_minibatch_loss)
            d_real_minibatch_list.append(d_real_minibatch_loss)
            d_gradpen_minibatch_list.append(d_gradpen_minibatch_loss)

        d_gen_average_file.write(str(average_lst(d_gen_minibatch_list)) + "\n")
        d_fake_average_file.write(str(average_lst(d_fake_minibatch_list)) + "\n")
        d_real_average_file.write(str(average_lst(d_real_minibatch_list)) + "\n")
        d_gradpen_average_file.write(str(average_lst(d_gradpen_minibatch_list)) + "\n")

        generator_loss = generator_model.train_on_batch(np.random.rand(BATCH_SIZE, LATENT_SIZE), positive_y)
        g_file.write(str(generator_loss) + "\n")
    generate_images(generator, output_dir, epoch, False)
    generate_images(generator, output_dir, epoch, True)

    if (epoch + 1) < 300:
        if (epoch + 1)%50 == 0:
            generator.save(weights_dir + 'generator_{}_{}_{}_{}.h5'.format(TYPE, BATCH_SIZE, LATENT_SIZE, (epoch + 1)))
    elif (epoch + 1) < 600:
        if (epoch + 1)% 5 == 0:
            generator.save(weights_dir + 'generator_{}_{}_{}_{}.h5'.format(TYPE, BATCH_SIZE, LATENT_SIZE, (epoch + 1)))
    else:
        if (epoch + 1)%50 == 0:
            generator.save(weights_dir + 'generator_{}_{}_{}_{}.h5'.format(TYPE, BATCH_SIZE, LATENT_SIZE, (epoch + 1)))


