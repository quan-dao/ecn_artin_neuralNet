from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


mndata = MNIST('./data/train', return_type='numpy')
mndata.gz = True
images, labels = mndata.load_training()

# Info of images:
print('type : ', type(images))
print('shape: ', images.shape)
# type: numpy.ndarray
# shape: (60000,784) => a row stored a flatten image

# Check out 1 image
# print('size: ', images[0].shape)
# img = images[0].reshape((28, 28))
# print(img)
# plt.imshow(img, cmap='gray')
# plt.show()

# Normalize images
images /= 255
print('max: ', np.amax(images), ' min: ', np.amin(images))

