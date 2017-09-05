import Queue

class EnvironmentSimulator(object):
	def __init__(self, locations=[], funds=100.0, btc=0.0):

		self._original_funding = funds
		self._original_btc = btc

		self._locations = locations
		self._available_funds = funds
		self._available_btc = btc
		self._pnl = 0.0
		self.trades = Queue.Queue()
		self.pnls = [0.0]

	def reinit(self):
		self._available_funds = self._original_funding
		self._available_btc = self._original_btc
		self._pnl = 0.0
		self.trades = Queue.Queue()
		self.pnls = [0.0]		

	@property
	def locations(self):
		return self._locations

	@locations.setter
	def locations(self, x):
		self._locations = x

	@property
	def available_funds(self):
		return self._available_funds

	@available_funds.setter
	def available_funds(self, x):
		self._available_funds = x

	@property
	def available_btc(self):
		return self._available_btc

	@available_btc.setter
	def available_btc(self, x):
		self._available_btc = x

	@property
	def pnl(self):
		return self._pnl

	@pnl.setter
	def pnl(self, x):
		self._pnl = x

	def add_to_trades(self, time, size, price):
		self.trades.put({'time': time, 'size': size, 'price': price})

	def add_to_pnls(self, pnl):
		self.pnls.append(pnl)
