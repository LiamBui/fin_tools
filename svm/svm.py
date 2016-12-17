import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class SVM(object):

	def __init__(self, kernel, xTr, yTr):
		self.kernel = kernel
		self.xTr = xTr
		self.yTr = yTr
		self.model = None

	def train(self):
		if self.kernel == 'linear' or self.kernel == 'poly' or self.kernel == 'rbf' or self.kernel == 'sigmoid':
			self.model = svm.SVC(kernel=self.kernel)
		else:
			print 'Invalid SVM type : {1}\n'.format(self.kernel)
		self.model.fit(self.xTr, self.yTr)

	def predict(self, xTe):
		return self.model.predict(xTe)

	def score(self, xTe, yTe):
		return self.model.score(xTe, yTe)
