import pandas as pd
import matplotlib.pyplot as plt
import tools, datetime
import Queue as queue
import numpy as np
import mpld3

EPSILON_UPPER = 0.5
EPSILON_LOWER = 0.1
EPSILON_TILT_U = 0.7
EPSILON_TILT_L = 0.3
STOPPER = 0.99
SELL_POINT = 1.00
TIME_WINDOW = '1min'
FUNDS = 100.0
WAIT_TIME = datetime.timedelta(minutes=15)
LOCATION = '../data/btc/20170407.csv' #'../data/output.log' 

def get_data():
	df = pd.read_csv(LOCATION)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['datetime','tilt','price_delta','bid_size','bid','midprice','ask','ask_size']
	df['tilt_up'] = df['tilt'] > 0.5
	#df.columns = ['datetime','tilt','tilt_up','price_delta','bid','ask','midprice']

	return df

def graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations):

	xs = [x[0] for x in indicators_buy]
	ys = [x[1] for x in indicators_buy]

	xs1 = [x[0] for x in indicators_sell]
	ys1 = [x[1] for x in indicators_sell]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['midprice'], color='black', label='MidPrice')
	buy = ax1.scatter(xs, ys, color='green', label='Buy indicators')
	sell = ax1.scatter(xs1, ys1, color='red', label='Sell indicators')


	tooltip = mpld3.plugins.PointLabelTooltip(buy, labels=buy_annotations)
	mpld3.plugins.connect(fig, tooltip)
	tooltip2 = mpld3.plugins.PointLabelTooltip(sell, labels=sell_annotations)
	mpld3.plugins.connect(fig, tooltip2)

	#plt.legend(loc='upper left')
	#plt.ylabel('Price (USD)')
	#plt.xlabel('Time')
	#plt.show()


	mpld3.show()

def run(df, indicators_buy, indicators_sell, funds=1000.0):
	(idxs_buy, ys_buy) = zip(*indicators_buy)
	(idxs_sell, ys_sell) = zip(*indicators_sell)
	idxs_buy, idxs_sell, ys_buy, ys_sell = list(idxs_buy), list(idxs_sell), list(ys_buy), list(ys_sell)

	available_btc = 0.0 
	pnl = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask']
	tilts_buy = []
	tilts_sell = []

	for idx, row in df.iterrows():
		if funds > row['ask'] * 0.01:
			if idx > last_buy + WAIT_TIME:
				if idx in idxs_buy:
					tilts_buy.append(row['tilt'])
					trades.put({'buy': True, 'time': idx, 'price': row['ask'], 'size': 0.01})
					last_buy = idx
					last_price = row['ask']
					available_btc += 0.01
					funds -= row['ask'] * 0.01
					pnl -= row['ask'] * 0.01
		if available_btc > 0.0:
			if idx in idxs_sell or (row['bid'] <= last_price - STOPPER) or (row['bid'] >= last_price + SELL_POINT):
				tilts_sell.append(row['tilt'])
				if idx not in idxs_sell: 
					idxs_sell.append(idx)
					ys_sell.append(row['midprice'])
				trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
				available_btc -= 0.01
				funds += row['bid'] * 0.01
				pnl += row['bid'] * 0.01

	# print 'PnL : ${:.2f}\t Funds : ${:.2f}\t Available BTC : {:.8f}'.format(pnl, funds, available_btc)

	return trades, zip(idxs_sell, ys_sell), tilts_buy, tilts_sell

def get_indicators(df, buy_ratio, buy_tilt, sell_ratio, sell_tilt):
	indicators_buy = []
	indicators_sell = []

	df = df.sort_index()
	rolling = df['tilt_up'].rolling(TIME_WINDOW, min_periods=1)
	df['indicator_buy'] = ((rolling.sum() / rolling.count()) >= buy_ratio) & (df['tilt'] >= buy_tilt)
	df['indicator_sell'] = ((rolling.sum() / rolling.count()) <= sell_ratio) & (df['tilt'] <= sell_tilt)

	# for idx, row in df.iloc[25:].iterrows():
	# 	if row['tilt'] >= 0.9 and df.iloc[idx - 25 : idx]['tilt_up'].sum() >= 25:
	# 		indicators.append((df.iloc[idx]['datetime'], df.iloc[idx]['midprice']))
	for idx, row in df.iterrows():
		if row['indicator_buy']:
			indicators_buy.append((row['datetime'], row['midprice']))
		elif row['indicator_sell']:
			indicators_sell.append((row['datetime'], row['midprice']))
		else:
			pass
	return indicators_buy, indicators_sell

def main():
	df = get_data()
	# for i in np.arange(0.7,0.81,0.05):
	# 	for j in np.arange(0.20, 0.31, 0.05):
	# 		for k in np.arange(0.7,0.81,0.05):
	# 			for l in np.arange(0.20,0.31,0.05):
	# 				EPSILON_UPPER = i
	# 				EPSILON_LOWER = j
	# 				EPSILON_TILT_U = k
	# 				EPSILON_TILT_L = l
	# 				print 'UPPER: {}\t LOWER: {}\t TILT_U: {}\t TILT_L: {}'.format(EPSILON_UPPER, EPSILON_LOWER, EPSILON_TILT_U, EPSILON_TILT_L)
	# 				indicators_buy, indicators_sell = get_indicators(df, EPSILON_TILT_U, EPSILON_UPPER, EPSILON_TILT_L, EPSILON_LOWER)
	# 				print 'Number of buy indicators: {}\t Number of sell indicators: {}'.format(len(indicators_buy), len(indicators_sell))
	# 				trades, indicators_sell, tilts_buy, tilts_sell= run(df, indicators_buy, indicators_sell, funds=FUNDS)
	# 				price_deltas = tools.compute_statistics(trades, funds=FUNDS)
	# 				tools.compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=FUNDS)
	indicators_buy, indicators_sell = get_indicators(df, EPSILON_TILT_U, EPSILON_UPPER, EPSILON_TILT_L, EPSILON_LOWER)
	print 'Number of buy indicators: {}\t Number of sell indicators: {}'.format(len(indicators_buy), len(indicators_sell))
	trades, indicators_sell, tilts_buy, tilts_sell= run(df, indicators_buy, indicators_sell, funds=FUNDS)
	price_deltas = tools.compute_statistics(trades, funds=FUNDS)
	tools.compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=FUNDS)
	graph(df, indicators_buy, indicators_sell, tilts_buy, tilts_sell)


if __name__ == '__main__':
	main()