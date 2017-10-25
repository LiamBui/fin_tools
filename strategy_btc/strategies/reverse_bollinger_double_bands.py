import pandas as pd
import math, datetime
from shared import funds_round, update_avg_entry, update_pnl, get_data

class ReverseExpDualBandStrategy(object):

	def __init__(self, environment, trader, epsilon_lower=-0.6, epsilon_upper=1.0, lookback=3600,
		         halflife=600, wait_sell=datetime.timedelta(minutes=20), wait_buy=datetime.timedelta(minutes=5),
		         wait_puke=datetime.timedelta(minutes=1), epsilon_puke=2.0,
		         epsilon_resistance=5.0, epsilon_support=5.0,
		         verbose=False,
				 halflife2=1800):

		self.environment = environment
		self.trader = trader
		self.epsilon_lower = epsilon_lower
		self.epsilon_upper = epsilon_upper
		self.lookback = lookback
		self.halflife = halflife
		self.halflife2 = halflife2
		self.wait_sell = wait_sell
		self.wait_buy = wait_buy
		self.wait_puke = wait_puke
		self.epsilon_puke = epsilon_puke
		self.epsilon_resistance = epsilon_resistance
		self.epsilon_support = epsilon_support
		self.verbose = verbose
		self.sell_indicators = []
		self.buy_indicators = []
		self.sell_annotations = []
		self.buy_annotations = []

	def __str__(self):
		return '''Reverse Exponential Strategy with following parameters: 
		Epsilon upper: \t {:.2f} 
		Epsilon lower: \t {:.2f} 
		Half-life: \t {:.2f} minutes
		Look back: \t {:.2f} minutes
		Wait to sell: \t {} 
		Wait to buy: \t {} 
		Wait to puke: \t {} 
		Epsilon puke: \t {:.2f} 
		Epsilon resistance (optional): \t {:.2f} 
		Epsilon support (optional): \t {:.2f}'''.format(self.epsilon_upper, self.epsilon_lower, 
			self.halflife / 60, self.lookback / 60, self.wait_sell, self.wait_buy, self.wait_puke, 
			self.epsilon_puke, self.epsilon_resistance, self.epsilon_support)

	def initialize_stats_exp(self, df, halflife=600):

		rolling = df['mid_price'].ewm(halflife=halflife, min_periods=600)
		mean = rolling.mean()
		var = rolling.var()
		std = rolling.std()
		return mean.tail(n=1).iloc[0], var.tail(n=1).iloc[0], std.tail(n=1).iloc[0]

	def update_stats(self, new, mean, var, halflife=600):

		alpha = 1.0 - math.exp(math.log(0.5)/float(halflife))
		diff = new - mean
		incr = alpha * diff
		mean += incr
		var = (1.0 - alpha) * (var + diff * incr)
		std = math.sqrt(var)
		return mean, var, std

	def init_df(self, df):

		self.last_buy = df.iloc[0]['datetime']
		self.last_sell = df.iloc[0]['datetime']
		self.last_price = df.iloc[0]['ask_price']
		self.avg_entry = df.iloc[0]['ask_price']

		df_init = df.head(n=self.lookback)
		df = df.tail(n=(len(df.index)-self.lookback))
		self.mean, self.var, self.std = self.initialize_stats_exp(df_init, halflife=self.halflife)

		return df.iloc[self.lookback:]

	def buy(self, idx, size_buy, price_buy):

		if self.trader.buy_fill(size_buy, price_buy):
			self.avg_entry = update_avg_entry(self.avg_entry, self.environment.available_btc, price_buy, size_buy)
			self.environment.available_btc += size_buy
			self.environment.available_funds -= price_buy * size_buy
			self.last_buy = idx
			self.environment.add_to_trades(idx, size_buy, price_buy)

			self.buy_indicators.append((idx, price_buy))
			self.buy_annotations.append(price_buy)

			if self.verbose:
				print '{}: Bought {:.8f} @ {:.2f}. avg_entry: {:.2f}'.format(idx, size_buy, price_buy, self.avg_entry)


	def sell(self, idx, size_sell, price_sell):

		if self.trader.sell_fill(size_sell, price_sell):
			self.environment.pnl = update_pnl(self.environment.pnl, self.avg_entry, price_sell, size_sell) - 0.04
			self.environment.add_to_pnls(self.environment.pnl)
			self.environment.available_btc -= size_sell
			self.environment.available_funds += size_sell * price_sell - 0.04
			self.last_sell = idx
			self.environment.add_to_trades(idx, -size_sell, price_sell)
			self.environment.add_to_returns(self.environment.available_funds + self.environment.available_btc * price_sell)

			self.sell_indicators.append((idx, price_sell))
			self.sell_annotations.append(price_sell)

			if self.verbose:
				print '{}: Sold {:.8f} @ {:.2f}. pnl: {:.2f}'.format(idx, size_sell, price_sell, self.environment.pnl)

	def run(self):

		if self.verbose:
			print 'Running basic reverse bollinger band strategy.'

		for location in self.environment.locations:
			df = get_data(location)

			df = self.init_df(df)

			for idx, row in df.iterrows():

				self.mean, self.var, self.std = self.update_stats(row['mid_price'], self.mean, self.var, halflife=self.halflife)
				self.mean2, self.var2, self.std2 = self.update_stats(row['mid_price'], self.mean2, self.var2, halflife=self.halflife2)

				upper_band = self.mean2 + self.epsilon_upper * self.std2
				lower_band = self.mean - self.epsilon_lower * self.std
				resistance_band = self.mean + self.epsilon_resistance * self.std
				support_band = self.mean - self.epsilon_support * self.std
				stoppage_amt = self.epsilon_puke * self.std

				if row['num'] < 60:
					continue

				if row['mid_price'] != df.iloc[df.index.get_loc(idx) - 1]['mid_price']:

					price_buy = funds_round(row['mid_price'], direction=math.floor)
					price_sell = funds_round(row['mid_price'], direction=math.ceil)

					# Attempt to buy
					funds_required = price_buy * 0.01
					if self.environment.available_funds > funds_required:

						size_buy = self.environment.available_funds / price_buy
						if row['mid_price'] > upper_band and idx > self.last_sell + self.wait_buy:
							self.buy(idx, size_buy, price_buy)

					if self.environment.available_btc > 0:

						size_sell = self.environment.available_btc
						if row['mid_price'] < lower_band and idx > self.last_buy + self.wait_sell:
							self.sell(idx, size_sell, price_sell)
						# elif row['mid_price'] < self.avg_entry - stoppage_amt and idx > self.last_buy + self.wait_puke:
						# 	self.sell(idx, size_sell, price_sell)

			if self.verbose:
				print 'Done. Final pnl: {:.2f}'.format(self.environment.pnl)

	def run_with_support_resistance(self):

		if self.verbose:
			print 'Running reverse bollinger band strategy with support and resistance.'

		for location in self.environment.locations:
			df = get_data(location)

			df = self.init_df(df)

			for idx, row in df.iterrows():

				self.mean, self.var, self.std = self.update_stats(row['mid_price'], self.mean, self.var, halflife=self.halflife)

				upper_band = self.mean + self.epsilon_upper * self.std
				lower_band = self.mean - self.epsilon_lower * self.std
				resistance_band = self.mean + self.epsilon_resistance * self.std
				support_band = self.mean - self.epsilon_support * self.std
				stoppage_amt = self.epsilon_puke * self.std

				if row['num'] < 60:
					continue

				if row['mid_price'] != df.iloc[df.index.get_loc(idx) - 1]['mid_price']:

					price_buy = funds_round(row['mid_price'], direction=math.floor)
					price_sell = funds_round(row['mid_price'], direction=math.ceil)

					# Attempt to buy
					funds_required = price_buy * 0.01
					if self.environment.available_funds > funds_required:

						size_buy = self.environment.available_funds / price_buy
						if row['mid_price'] > upper_band and row['mid_price'] < resistance_band and idx > self.last_sell + self.wait_buy:
							self.buy(idx, size_buy, price_buy)

					if self.environment.available_btc > 0:

						size_sell = self.environment.available_btc
						if row['mid_price'] < lower_band and row['mid_price'] > support_band and idx > self.last_buy + self.wait_sell:
							self.sell(idx, size_sell, price_sell)
						elif row['mid_price'] < self.avg_entry - stoppage_amt and idx > self.last_buy + self.wait_puke:
							self.sell(idx, size_sell, price_sell)

			if self.verbose:
				print 'Done. Final pnl: {:.2f}'.format(self.environment.pnl)
