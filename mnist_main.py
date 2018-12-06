from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from neuralNetClass import NeuralNet


mndata = MNIST('./data', return_type='numpy')
mndata.gz = True
train_images, _train_labels = mndata.load_training()
test_images, _test_labels = mndata.load_testing()

# Normalize images
train_images = train_images * 1.0 / 255
test_images = test_images * 1.0 / 255
# train_images /= 255
# test_images /= 255

# One-hot encoding
train_labels = np.zeros((_train_labels.shape[0], 10))
for i in range(_train_labels.shape[0]):
	train_labels[i, _train_labels[i]] = 1
print 'Train data is loaded'

test_labels = np.zeros((_test_labels.shape[0], 10))
for i in range(_test_labels.shape[0]):
	test_labels[i, _test_labels[i]] = 1
print 'Test data is loaded'

img_size = 28 * 28
batch_size = 10
learning_rate = 0.5
model = NeuralNet(img_size, 10, 1, [30], batch_size, learning_rate, 0.1, 0.1)

epoch = 0
max_epoch = 10
while epoch < max_epoch:
	# shuffle the training set
	_index = np.random.permutation(train_images.shape[0])
	for _s in xrange(0, train_images.shape[0], batch_size):  # _index.shape[0]
		# print _s, '---------', _s + batch_size
		batch_train_images = train_images[_index[_s : _s + batch_size], :].transpose()
		batch_train_labels = train_labels[_index[_s : _s + batch_size], :].transpose()
		# Forward, then Backward propagation
		model.netFedInput(batch_train_images)
		mod_err = model.netBackProp(batch_train_labels)
		model.netWeightUpdate()

		
	# Test the model
	correct = model.netEval(test_images.transpose(), test_labels.transpose())
	print "Epoch %d\t Correct guess: %f%%" % (epoch, correct*100)

	#increase epocjh
	epoch += 1
	print '***********************************************************'

#Restore weights
model.netWeightsRestore()

correct = model.netEval(test_images.transpose(), test_labels.transpose())
print "End model\t Correct guess: %f%%" % (correct*100)