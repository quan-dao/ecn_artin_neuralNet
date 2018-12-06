import numpy as np


class NeuralNet(object):
	"""docstring for NeuralNet"""
	def __init__(self, num_input_layer, num_output_layer, num_hidden_layers, list_num_neurons_per_hidden, batch_size=1, _learning_rate=0.01, _l2_regularization=0.0, _drop_prob = 0.0):
		self.n_in = num_input_layer  # number of neurons on the input layer
		self.n_out = num_output_layer  # number of neurons on the output layer
		self.depth = num_hidden_layers + 2  # total number of layers in the model
		self.learning_rate = _learning_rate

		# create and initialize weight matrix & bias vector for every layer
		self.weights_dict = {}
		self.biases_dict = {}

		for i, num_neurons in enumerate(list_num_neurons_per_hidden):
			if i == 0:
				sigma = 1 / np.sqrt(self.n_in)
				self.weights_dict[1] = np.random.normal(0, sigma, (num_neurons, self.n_in))  # input -> first hidden layer
			else:
				sigma = 1 / np.sqrt(list_num_neurons_per_hidden[i - 1])
				self.weights_dict[i + 1] = np.random.normal(0, sigma, (num_neurons, list_num_neurons_per_hidden[i - 1]))  # hidden i - 1 -> hidden i
			self.biases_dict[i + 1] = np.zeros((num_neurons, 1))
		# last hidden -> output
		self.weights_dict[self.depth - 1] = np.random.normal(0, 1 / np.sqrt(list_num_neurons_per_hidden[-1]), (self.n_out, list_num_neurons_per_hidden[-1]))  
		self.biases_dict[self.depth - 1] = np.zeros((self.n_out, 1))

		# storing dictionary
		self.weighted_input_dict = {}  # store forward propagation result
		self.modified_error_dict ={}  # store backward propagation result
		self.net_input = np.zeros((self.n_in, batch_size))

		# Add-on param
		self.l2_regu = _l2_regularization
		self.keep_prob = 1.0 - _drop_prob
		self.cut_list = []



	def layerAct(self, x):
		return 1 / (1 + np.exp(-x))


	def layerDiffAct(self, x):
		return self.layerAct(x) * (1 - self.layerAct(x))


	def netFedInput(self, _net_input):
		assert _net_input.shape[0] == self.n_in
		self.net_input = np.zeros(_net_input.shape) + _net_input


	def netForwardProp(self):
		# reset cut_list
		self.cut_list = []
		# Go through weights dictionary and keep the connection with probability self.keep_prob
		for key in self.weights_dict.keys():
			for i in range(self.weights_dict[key].shape[0]):
				keep = np.random.uniform(0, 1) < self.keep_prob
				if not keep:
					# set this row to zeros
					self.weights_dict[key][i, :] = 0
					self.cut_list.append(i)

		m = self.net_input.shape[1]
		vec_one = np.zeros((1, m)) + 1 
		for key in self.weights_dict.keys():
			if key == 1:
				z = np.matmul(self.weights_dict[key], self.net_input) + self.biases_dict[key]
			else:
				z = np.matmul(self.weights_dict[key], 
					self.layerAct(self.weighted_input_dict[key - 1])) + self.biases_dict[key]
			# store z
			self.weighted_input_dict[key] = z 

		# return np.exp(self.weighted_input_dict[self.depth - 1]) / np.sum((np.exp(self.weighted_input_dict[self.depth - 1])), axis=0)  #softmax
		return self.layerAct(self.weighted_input_dict[self.depth - 1])


	def netWeightsRestore(self):
		for key in self.weights_dict.keys():
			self.weights_dict[key] *=  self.keep_prob


	def netBackProp(self, true_labels):
		self.modified_error_dict[self.depth - 1] = (self.netForwardProp() - true_labels)
		for i in xrange(self.depth - 2, 0, -1):
			self.modified_error_dict[i] = np.matmul(self.weights_dict[i + 1].transpose(), 
				self.modified_error_dict[i + 1]) * self.layerDiffAct(self.weighted_input_dict[i])

		return self.modified_error_dict


	def netWeightUpdate(self):
		alpha = self.learning_rate
		m = self.net_input.shape[1]
		vec_one = np.zeros((1, m)) + 1
		for i in range(1, self.depth):
			# Normal loss update
			if i not in self.cut_list:
				if i == 1:
					self.weights_dict[i] -= (alpha / m) * np.matmul(self.modified_error_dict[i], self.net_input.transpose()) + \
					(self.l2_regu / m) * self.weights_dict[i]  # L2 regularization loss
				else:
					self.weights_dict[i] -= (alpha / m) * np.matmul(self.modified_error_dict[i], self.layerAct(self.weighted_input_dict[i - 1]).transpose()) + \
					(self.l2_regu / m) * self.weights_dict[i]  # L2 regularization loss
				# update biases
				self.biases_dict[i] -= (alpha / m) * np.matmul(self.modified_error_dict[i], vec_one.transpose())


	def netEval(self, test_data, test_label):
		self.netFedInput(test_data)
		predict = self.netForwardProp()
		# convert predict to binary
		predict = predict >= np.amax(predict, axis=0)
		# count the wrong prediction
		diff_matrix = predict.astype(int) - test_label
		num_wrong = np.sum((diff_matrix == -1).astype(int))
		# print "num_wrong = ", num_wrong
		# print "num test = ", test_label.shape
		return 1 - num_wrong * 1.0 / test_label.shape[1]

		
if __name__ == '__main__':
	test_ann = NeuralNet(3, 2, 1, [3])
	for key in test_ann.weights_dict.keys():
		print("weights matrix ", key)
		print(test_ann.weights_dict[key])
		print("biases vector ")
		print(test_ann.biases_dict[key])
		print("-----")
	
	net_input = np.eye(3)
	true_labels = np.array([[1, 0], [0, 1], [1, 1]])

	test_ann.netFedInput(net_input)
	
	mod_error = test_ann.netBackProp(true_labels.transpose())
	print '------------- Modified Error -----------------'
	for key in mod_error.keys():
		print 'Key = ', key
		print mod_error[key]
	
	test_ann.netWeightUpdate()
	print '------------ Updated weights ---------'

	for key in test_ann.weights_dict.keys():
		print("weights matrix ", key)
		print(test_ann.weights_dict[key])
		print("biases vector ")
		print(test_ann.biases_dict[key])
		print("-------------------")
	
	correct = test_ann.netEval(net_input, true_labels.transpose())

	print("Output: ", correct * 100)
