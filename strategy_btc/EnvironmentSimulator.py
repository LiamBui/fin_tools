import Queue

class EnvironmentSimulator(object):
	def __init__(self, locations=[], funds=100.0, btc=0.0):
		self.locations = locations
		self.available_funds = funds
		self.available_btc = btc
		self.pnl = 0.0
		self.trades = Queue.Queue()

	@property
	def locations(self):
		return self.locations

	@locations.setter
	def locations(self, x):
		self.locations = x

	@property
	def available_funds(self):
		return self.available_funds

	@available_funds.setter
	def available_funds(self, x):
		self.available_funds = x

	@property
	def available_btc(self):
		return self.available_btc

	@available_btc.setter
	def available_btc(self, x):
		self.available_btc = x

	@property
	def trades(self)
		return self.trades

	@property
	def pnl(self):
		return self.pnl