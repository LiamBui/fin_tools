import random

'''
Initialize with slippage_type (distribution type) and slippage_params (depending on distribution chosen). 
Idea is that given the slippage type provided, we can more accurately determine if a trade will go through or not.
'''
class TradeSimulator(object):

	def __init__(self, slippage_type='static', slippage_params={p: 1.0}):

		# todo: slippage types
		self.slippage_type = slippage_type
		if self.slippage_type != 'static' and self.slippage_params == {p: 1.0}:
			print 'Must init TradeSimulator with slippage_params for given slippage type.'

		'''
		Static slippage. Fixed probability of order fill defined by 'p' in slippage_params.
		'''	
		if self.slippage_type.lower() == 'static':
			if 'p' not in self.slippage_params:
				print "slippage_params must include 'p', the fixed probability of order fill."
				break
			self.pdf = lambda size, price: self.slippage_params['p']
			self.cdf = None
			
		'''
		Uniformly distributed slippage. Probability of order fill defined by 'a' and 'b' in slippage_params.
		f(x) = 1 / (b-a)
		F(x) = (x-a) / (b-a)
		'''
		elif slippage_type.lower() == 'uniform':
			self.slippage_params = slippage_params
			elif 'a' not in self.slippage_params or 'b' not in self.slippage_params:
				print "slippage_params must include 'a' and 'b', the lower and upper bounds. Distribution: unif(a,b)."
				break
			def pdf(size, price):
				if price <= self.slippage_params['b'] and price >= self.slippage_params['a']:
					try:
						return 1.0 / (self.slippage_params['b'] - self.slippage_params['a'])
					except:
						return None
				else:
					return 0
			def cdf(size, price):
				if price < self.slippage_params['a']:
					return 0
				elif price > self.slippage_params['b']:
					return 1
				else:
					try:
						return (price - self.slippage_params['a']) / (self.slippage_params['b'] - self.slippage_params['a'])
					except:
						return None
			self.pdf = pdf
			self.cdf = cdf
		else:
			print 'Slippage type not valid.'
			break

	@property
	def pdf(self):
		return self.pdf

	@pdf.setter
	def pdf(self, f):
		self.pdf = f

	@property
	def cdf(self):
		return self.cdf

	@cdf.setter
	def cdf(self, f):
		self.cdf = f

	def buy_fill(self, size, price):
		if random.random() < self.pdf(size, price):
			return True
		else:
			return False

	def sell_fill(self, size, price):
		if random.random() < self.pdf(size, price):
			return True
		else:
			return False