from scipy import optimize

class Trainer(object):
	def __init__(self, Neural_Network):
		self.NN = Neural_Network

	def callback_weight(self, params):
		self.NN.set_params_weight(params)

	def loss_weight_wrapper(self, params, x, y):
		self.NN.set_params_weight(params)
		loss, grad = self.NN.loss(x, y), self.NN.compute_gradients(x, y)[0]
		return loss, grad

	def callback_bias(self, params):
		self.NN.set_params_bias(params)

	def loss_bias_wrapper(self, params, x, y):
		self.NN.set_params_bias(params)
		loss, grad = self.NN.loss(x, y), self.NN.compute_gradients(x, y)[1]
		return loss, grad

	def train(self, xTr, yTr, xTe, yTe):
		self.xTr = xTr
		self.yTr = yTr
		self.xTe = xTe
		self.yTe = yTe

		params_weight = self.NN.get_params_weight()
		params_bias = self.NN.get_params_bias()

		options = {'maxiter': 18}
		params_weight_prime = optimize.minimize(self.loss_weight_wrapper, params_weight, jac=True, method='CG', args=(xTr, yTr), options=options, callback=self.callback_weight)
		params_bias_prime = optimize.minimize(self.loss_bias_wrapper, params_bias, jac=True, method='CG', args=(xTr, yTr), options=options, callback=self.callback_bias)

		self.NN.set_params_weight(params_weight_prime.x)
		self.NN.set_params_bias(params_bias_prime.x)


