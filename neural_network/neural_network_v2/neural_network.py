import numpy as np
import trans_func

class Neural_Network(object):

	'''
	n must be equal to the number of rows (examples) of data.

	Size must be equal to a 3-tuple containing the input layer size, the hidden layer size, and the output layer size (number of nodes), in that order.
	
	f must be equal to the activation function type, as a string (e.g., sigmoid, tanh, ReLU, etc.)
	'''
	def __init__(self, n, size, f, regularization=0):
		self.inputlayersize, self.hiddenlayersize, self.outputlayersize = size
		self.n = n
		self.f, self.f_der = trans_func.get_trans_func(f)
		self.lmbd = regularization

		self.w1 = np.random.randn(self.inputlayersize, self.hiddenlayersize)
		self.w2 = np.random.randn(self.hiddenlayersize, self.outputlayersize)
		self.b1 = np.random.randn(self.n, self.hiddenlayersize)
		self.b2 = np.random.randn(self.n, self.outputlayersize)

	def forward(self, x):
		try:
			self.net2 = np.dot(x, self.w1) + self.b1
			self.y2 = self.f(self.net2)
			self.net3 = np.dot(self.y2, self.w2) + self.b2
		except:
			self.net2 = np.dot(x, self.w1)
			self.y2 = self.f(self.net2)
			self.net3 = np.dot(self.y2, self.w2)
		yhat = self.f(self.net3)
		return yhat

	def loss(self, x, y):
		self.yhat = self.forward(x)
		return 0.5 * np.sum((np.power(y - self.yhat, 2))) / x.shape[0] + (self.lmbd / 2) * np.sum(np.power(self.w1, 2)) + np.sum(np.power(self.w2, 2))

	def backprop(self, x, y):
		self.yhat = self.forward(x)

		delta3 = np.multiply(-(y - self.yhat), self.f_der(self.net3))
		dCostdw2 = np.dot(self.y2.T, delta3) + self.lmbd * self.w2
		try:
			dCostdb2 = delta3.as_matrix()
		except:
			dCostdb2 = delta3

		delta2 = np.dot(delta3, self.w2.T) * self.f_der(self.net2)
		dCostdw1 = np.dot(x.T, delta2) + self.lmbd * self.w1
		dCostdb1 = delta2

		return dCostdw2, dCostdb2, dCostdw1, dCostdb1

	'''
	Helper methods for the Trainer class.
	'''
	def set_params_weight(self, params):
		w1_start, w1_end = 0, self.hiddenlayersize * self.inputlayersize
		self.w1 = np.reshape(params[w1_start : w1_end], (self.inputlayersize, self.hiddenlayersize))
		w2_end = w1_end + self.hiddenlayersize * self.outputlayersize
		self.w2 = np.reshape(params[w1_end : w2_end], (self.hiddenlayersize, self.outputlayersize))

	def set_params_bias(self, params):
		b1_start = 0
		b1_end = self.n * self.hiddenlayersize
		self.b1 = np.reshape(params[b1_start : b1_end], (self.n, self.hiddenlayersize))
		b2_end = b1_end + self.n * self.outputlayersize
		self.b2 = np.reshape(params[b1_end : b2_end], (self.n, self.outputlayersize))

	def get_params_weight(self):
		params = np.concatenate((self.w1.ravel(), self.w2.ravel()))
		return params

	def get_params_bias(self):
		params = np.concatenate((self.b1.ravel(), self.b2.ravel())) 
		return params

	def compute_gradients(self, x, y):
		dCostdw2, dCostdb2, dCostdw1, dCostdb1 = self.backprop(x, y)
		return np.concatenate( (dCostdw1.ravel(), dCostdw2.ravel()) ), np.concatenate( (dCostdb1.ravel(), dCostdb2.ravel()) )