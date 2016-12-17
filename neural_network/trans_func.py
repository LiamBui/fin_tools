import numpy as np
import sys

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
def sigmoid_der(x):
	return np.exp(-x) / ((1.0 + np.exp(-x))**2)

def relu(x):
	return np.maximum(x,0,x)
def relu_der(x):
	return 1.0 * (x > 0)

def tanh(x):
	return np.tanh(x)
def tanh_der(x):
	return 1.0 - (np.power(np.tanh(x), 2))

def relu2(x):
	return 0.5 * (np.power(np.maximum(x,0,x), 2))
def relu2_der(x):
	return x * (x > 0)

def sin(x):
	return np.sin(x)
def sin_der(x):
	return np.cos(x)

def get_trans_func(typ):
	if typ.lower() == 'sigmoid':
		return sigmoid, sigmoid_der
	elif typ.lower() == 'relu':
		return relu, relu_der
	elif typ.lower() == 'tanh':
		return tanh, tanh_der
	elif typ.lower() == 'relu2':
		return relu2, relu2_der
	elif typ.lower() == 'sin':
		return sin, sin_der
	else:
		print 'No such function.'
		sys.exit(2)