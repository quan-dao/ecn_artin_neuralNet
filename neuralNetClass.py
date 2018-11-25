import numpy as np


class NeuralNet(object):
	"""docstring for NeuralNet"""
	def __init__(self, num_input_layer, num_output_layer, num_hidden_layers, list_num_neurons_per_hidden, learning_rate=0.01):
		self.n_in = num_input_layer  # number of neurons on the input layer
		self.n_out = num_output_layer  # number of neurons on the output layer
		self.depth = num_hidden_layers + 2  # total number of layers in the model
		self.learning_rate = learning_rate

		# create and initialize weight matrix & bias vector for every layer
		self.weights_dict = {}
		self.biases_dict = {}
		for i, num_neurons in enumerate(list_num_neurons_per_hidden):
			sigma = 1 / np.sqrt(num_neurons - 1)
			if i == 0:
				self.weights_dict[1] = np.random.normal(0, sigma, (num_neurons, self.n_in))  # input -> first hidden layer
			else:
				self.weights_dict[i + 1] = np.random.normal(0, sigma, (num_neurons, list_num_neurons_per_hidden[i - 1]))  # hidden i - 1 -> hidden i
			self.biases_dict[i + 1] = np.zeros((num_neurons, 1))
		# last hidden -> output
		self.weights_dict[self.depth - 1] = np.random.normal(0, 1 / np.sqrt(self.n_out - 1), (self.n_out, list_num_neurons_per_hidden[-1]))  
		self.biases_dict[self.depth - 1] = np.zeros((self.n_out, 1))

		# storing dictionary
		self.weighted_input_dict = {}  # store forward propagation result
		self.modified_error_dict ={}  # store backward propagation result


	def layerAct(self, x):
		return 1 / (1 + np.exp(x))


	def layerDiffAct(self, x):
		return self.layerAct(x) * (1 - self.layerAct(x))


	def netForwardProp(self, net_input):
		for key in self.weights_dict.keys():
			if key == 1:
				z = np.matmul(self.weights_dict[key], net_input) + self.biases_dict[key]
			else:
				z = np.matmul(self.weights_dict[key], self.layerAct(self.weighted_input_dict[key - 1])) + self.biases_dict[key]
			# store z
			self.weighted_input_dict[key] = z 

		return self.layerAct(self.weighted_input_dict[self.depth - 1])


	def netBackProp(self, net_input, true_labels):
		self.modified_error_dict[self.depth - 1] = self.netForwardProp(net_input) - true_labels
		for i in xrange(self.depth - 2, 0, -1):
			self.modified_error_dict[i] = np.matmul(self.weights_dict[i + 1].transpose(), self.modified_error_dict[i + 1]) * self.layerDiffAct(self.weighted_input_dict[i])

		

if __name__ == '__main__':
	test_ann = NeuralNet(3, 2, 1, [2])
	for key in test_ann.weights_dict.keys():
		print("weights matrix ", key)
		print(test_ann.weights_dict[key])
		print("biases vector ")
		print(test_ann.biases_dict[key])
		print("-------------------")
	
	net_input = np.array([[1, 1, 1]])
	true_labels = np.array([[1, 0]])
	test_ann.netBackProp(net_input.transpose(), true_labels.transpose())
	print("Output: ")
	print(test_ann.modified_error_dict)
