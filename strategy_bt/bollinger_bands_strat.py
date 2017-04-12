import pandas as pd
import matplotlib.pyplot as plt
import tools, datetime
import Queue as queue
import numpy as np
import mpld3

LOOKBACK = '1h'
EPSILON_UPPER = 3
EPSILON_LOWER = 3
FUNDS = 100.0
STOPPER = 0.99
SELL_POINT = 1.00
WAIT_TIME = datetime.timedelta(minutes=20)
MAX_POSITION = 0.05
LOCATION = '../data/btc/20170408.csv' #'../data/output.log' 

def get_data():
	df = pd.read_csv(LOCATION)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['datetime','tilt','price_delta','bid_size','bid','midprice','ask','ask_size']
	rolling = df['midprice'].rolling(LOOKBACK, min_periods=1)
	df['moving_average'] = rolling.mean()
	df['moving_std'] = rolling.std()
	df['upper_band'] = df['moving_average'] + EPSILON_UPPER * df['moving_std']
	df['lower_band'] = df['moving_average'] - EPSILON_LOWER * df['moving_std']

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

	# plt.legend(loc='upper left')
	# plt.ylabel('Price (USD)')
	# plt.xlabel('Time')
	# plt.show()

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
	buy_annotations = []
	sell_annotations = []
	total_position = 0

	for idx, row in df.iterrows():
		if funds > row['ask'] * 0.01:
			if idx > last_buy + datetime.timedelta(seconds=15) and (idx in idxs_buy) and total_position < MAX_POSITION:
				total_position += 0.01
				buy_annotations.append(row['midprice'])
				trades.put({'buy': True, 'time': idx, 'price': row['ask'], 'size': 0.01})
				last_buy = idx
				last_price = row['ask']
				available_btc += 0.01
				funds -= row['ask'] * 0.01
				pnl -= row['ask'] * 0.01
		if available_btc > 0.0:
			if (row['bid'] <= last_price - STOPPER) and (idx > last_buy + datetime.timedelta(seconds=30)):
				total_position -= 0.01
				sell_annotations.append(row['midprice'])
				if idx not in idxs_sell: 
					idxs_sell.append(idx)
					ys_sell.append(row['midprice'])
				trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
				available_btc -= 0.01
				funds += row['bid'] * 0.01
				pnl += row['bid'] * 0.01
				continue
			if idx in idxs_sell and (idx > last_buy + WAIT_TIME):
				total_position -= 0.01
				sell_annotations.append(row['midprice'])
				if idx not in idxs_sell: 
					idxs_sell.append(idx)
					ys_sell.append(row['midprice'])
				trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
				available_btc -= 0.01
				funds += row['bid'] * 0.01
				pnl += row['bid'] * 0.01

	return trades, zip(idxs_sell, ys_sell), buy_annotations, sell_annotations

def get_indicators(df):
	indicators_buy = []
	indicators_sell = []

	df['indicator_buy'] = df['midprice'] < df['lower_band']
	df['indicator_sell'] = df['midprice'] > df['upper_band']

	for idx, row in df.iterrows():
		if row['indicator_buy']:
			indicators_buy.append((row['datetime'], row['midprice']))
		if row['indicator_sell']:
			indicators_sell.append((row['datetime'], row['midprice']))

	return indicators_buy, indicators_sell

def main():
	df = get_data()
	indicators_buy, indicators_sell = get_indicators(df)
	print 'Number of buy indicators: {}\t Number of sell indicators: {}'.format(len(indicators_buy), len(indicators_sell))
	trades, indicators_sell, buy_annotations, sell_annotations = run(df, indicators_buy, indicators_sell, funds=FUNDS)
	price_deltas = tools.compute_statistics(trades, funds=FUNDS)
	tools.compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=FUNDS)
	graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations)
	print 'done'


if __name__ == '__main__':
	main()

