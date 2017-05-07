import pandas as pd
import matplotlib.pyplot as plt
import tools, datetime
import Queue as queue
import numpy as np
import mpld3

LOOKBACK = '90min'
EPSILON_UPPER = 3
EPSILON_LOWER = 3
STOPPER = 5.00
WAIT_TIME = datetime.timedelta(minutes=15)
VOLATILITY_STOP = 7.50

FUNDS = 100.0
SELL_POINT = 2.00
MAX_POSITION = 0.10
STOP_EPSILON_UPPER = 3
STOP_EPSILON_LOWER = 3
STOP_LOOKBACK = '180min'

LOCATION = '../data/btc/20170418.csv' #'../data/output.log' 

def get_data():
	df = pd.read_csv(LOCATION)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['datetime','tilt','price_delta','bid_size','bid','midprice','ask','ask_size']

	rolling = df['midprice'].rolling(LOOKBACK, min_periods=1)
	stop_rolling = df['midprice'].rolling(STOP_LOOKBACK, min_periods=1)

	df['moving_average'] = rolling.mean()
	df['moving_std'] = rolling.std()
	df['upper_band'] = df['moving_average'] + EPSILON_UPPER * df['moving_std']
	df['lower_band'] = df['moving_average'] - EPSILON_LOWER * df['moving_std']

	# df['far_moving_avg'] = stop_rolling.mean()
	# df['far_moving_std'] = stop_rolling.std()
	# df['far_upper_band'] = df['far_moving_avg'] + STOP_EPSILON_UPPER * df['far_moving_std']
	# df['far_lower_band'] = df['far_moving_avg'] - STOP_EPSILON_LOWER * df['far_moving_std']

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

	# ax1.plot(df['datetime'], df['far_upper_band'], color='red', label='Far Upper')
	# ax1.plot(df['datetime'], df['far_lower_band'], color='red', label='Far Lower')
	
	ax1.plot(df['datetime'], df['upper_band'], color='blue', label='Upper')
	ax1.plot(df['datetime'], df['lower_band'], color='blue', label='Lower')

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
	# (idxs_buy, ys_buy) = zip(*indicators_buy)
	# (idxs_sell, ys_sell) = zip(*indicators_sell)
	# idxs_buy, idxs_sell, ys_buy, ys_sell = list(idxs_buy), list(idxs_sell), list(ys_buy), list(ys_sell)
	idxs_buy, ys_buy = [],[]
	idxs_sell, ys_sell = [],[]

	available_btc = 0.0 
	pnl = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask']
	buy_annotations = []
	sell_annotations = []
	total_position = 0
	last_loss = False
	average_entry = 0.0

	buy_streak = False
	sell_streak = False

	for idx, row in df.iterrows():

		# if row['midprice'] < row['far_lower_band']:
		# 	# FAR OUT OF THE MONEY; DO NOT BUY
		# 	print 'Out of the money @ {}'.format(idx)
		# 	continue

		# if row['midprice'] > row['far_upper_band']:
		# 	# FAR IN THE MONEY; LIQUIDATE
		# 	if available_btc > 0.0:
		# 		total_position -= available_btc
		# 		sell_annotations.append(row['midprice'])
		# 		if idx not in idxs_sell:
		# 			idxs_sell.append(idx)
		# 			ys_sell.append(row['midprice'])
		# 		trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
		# 		last_sell = idx
		# 		available_btc -= available_btc
		# 		funds += row['bid'] * available_btc
		# 		pnl += row['bid'] * available_btc - average_entry * available_btc

		# -------------------------------------------------------------------------------------------

		# if row['bid'] > row['upper_band'] and available_btc > 0.0:
		# 	# IN THE MONEY; LIQUIDATE
		# 	total_position -= available_btc
		# 	sell_annotations.append(row['midprice'])
		# 	if idx not in idxs_sell:
		# 		idxs_sell.append(idx)
		# 		ys_sell.append(row['midprice'])
		# 	trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
		# 	last_sell = idx
		# 	available_btc -= available_btc
		# 	funds += row['bid'] * available_btc
		# 	pnl += row['bid'] * available_btc - average_entry * available_btc
		# 	continue

		# if row['bid'] > last_price + SELL_POINT and available_btc > 0.0:
		# 	# IN THE MONEY; LIQUIDATE
		# 	total_position -= available_btc
		# 	sell_annotations.append(row['midprice'])
		# 	if idx not in idxs_sell:
		# 		idxs_sell.append(idx)
		# 		ys_sell.append(row['midprice'])
		# 	trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
		# 	last_sell = idx
		# 	available_btc -= available_btc
		# 	funds += row['bid'] * available_btc
		# 	pnl += row['bid'] * available_btc - average_entry * available_btc
		# 	continue

		# if row['bid'] < last_price - STOPPER and idx > last_buy + datetime.timedelta(minutes=1) and available_btc > 0.0:
		# 	# OUT OF THE MONEY; STOP
		# 	total_position -= 0.01
		# 	sell_annotations.append(row['midprice'])
		# 	if idx not in idxs_sell: 
		# 		idxs_sell.append(idx)
		# 		ys_sell.append(row['midprice'])
		# 	trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
		# 	last_sell = idx
		# 	available_btc -= 0.01
		# 	funds += row['bid'] * 0.01
		# 	if (row['bid'] * 0.01 - average_entry * 0.01) < -0.03:
		# 		last_loss = True
		# 	else:
		# 		last_loss = False
		# 	pnl += row['bid'] * 0.01 - average_entry * 0.01
		# 	continue

		# if idx in idxs_sell and (idx > last_buy + WAIT_TIME) and available_btc > 0.0:
		# 	# MAYBE IN THE MONEY OR OUT OF THE MONEY 
		# 	total_position -= 0.01
		# 	sell_annotations.append(row['midprice'])
		# 	if idx not in idxs_sell: 
		# 		idxs_sell.append(idx)
		# 		ys_sell.append(row['midprice'])
		# 	trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
		# 	last_sell = idx
		# 	available_btc -= 0.01
		# 	funds += row['bid'] * 0.01
		# 	if (row['bid'] * 0.01 - average_entry * 0.01) < -0.03:
		# 		last_loss = True
		# 	else:
		# 		last_loss = False
		# 	pnl += row['bid'] * 0.01 - average_entry * 0.01
		# 	continue

		# if row['upper_band'] - row['lower_band'] > VOLATILITY_STOP:
		# 	# IF TOO VOLATILE, DO NOT BUY 
		# 	continue

		# if funds > row['ask'] * 0.01:
		# 	# if (not last_loss) or idx > last_sell + datetime.timedelta(minutes=10):
		# 	if idx > last_sell + datetime.timedelta(minutes=10) and idx > last_buy + datetime.timedelta(seconds=15) and (idx in idxs_buy) and total_position < MAX_POSITION:
		# 		average_entry = (average_entry * available_btc + 0.01 * row['ask']) / (available_btc + 0.01)
		# 		total_position += 0.01
		# 		buy_annotations.append(row['midprice'])
		# 		trades.put({'buy': True, 'time': idx, 'price': row['ask'], 'size': 0.01})
		# 		last_buy = idx
		# 		last_price = row['ask']
		# 		available_btc += 0.01
		# 		funds -= row['ask'] * 0.01

		STOPPER = 2.0 * row['moving_std']
		SELL_POINT = 1.0 * row['moving_std']

		if available_btc > 0.0 and row['midprice'] > row['upper_band']:
			# IN THE MONEY; WAIT UNTIL WE'RE PEAKING
			sell_streak = True
			# continue
		if available_btc > 0.0 and sell_streak == True and idx > last_buy + WAIT_TIME:
			# IN THE MONEY; SELL STREAK COMING TO END
			total_position -= available_btc
			sell_annotations.append(row['midprice'])
			idxs_sell.append((idx, row['midprice']))
			trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
			last_sell = idx
			funds += row['bid'] * available_btc
			pnl += row['bid'] * available_btc - average_entry * available_btc
			available_btc -= available_btc
			sell_streak = False
			continue

		if available_btc > 0.0 and row['bid'] > average_entry + SELL_POINT:
			# IN THE MONEY; GET OUT
			sell_streak = False
			total_position -= available_btc
			sell_annotations.append(row['midprice'])
			idxs_sell.append((idx, row['midprice']))
			trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
			last_sell = idx
			funds += row['bid'] * available_btc
			pnl += row['bid'] * available_btc - average_entry * available_btc
			available_btc -= available_btc
			continue

		if available_btc > 0.0 and row['bid'] < average_entry - STOPPER and idx > last_buy + datetime.timedelta(minutes=1):
			# OUT OF THE MONEY; STOP
			total_position -= available_btc
			sell_annotations.append(row['midprice'])
			idxs_sell.append((idx, row['midprice']))
			trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
			last_sell = idx
			funds += row['bid'] * available_btc
			pnl += row['bid'] * available_btc - average_entry * available_btc
			available_btc -= available_btc
			sell_streak = False
			# continue

		# if row['upper_band'] - row['lower_band'] > VOLATILITY_STOP:
		# 	# IF TOO VOLATILE, DO NOT BUY 
		# 	if available_btc > 0.0:
		# 		sell_streak = False
		# 		total_position -= available_btc
		# 		sell_annotations.append(row['midprice'])
		# 		idxs_sell.append((idx, row['midprice']))
		# 		trades.put({'buy':False, 'time':idx, 'price': row['bid'], 'size': available_btc})
		# 		last_sell = idx
		# 		funds += row['bid'] * available_btc
		# 		pnl += row['bid'] * available_btc - average_entry * available_btc
		# 		available_btc -= available_btc	
		# 	continue

		if funds > row['ask'] * 0.01 and row['midprice'] < row['lower_band']:
			buy_streak = True
			# continue
		if funds > row['ask'] * 0.01 and buy_streak == True:
			if idx > last_sell + datetime.timedelta(minutes=10) and idx > last_buy + datetime.timedelta(seconds=15) and total_position < MAX_POSITION:
				average_entry = (average_entry * available_btc + 0.01 * row['ask']) / (available_btc + 0.01)
				total_position += 0.01
				buy_annotations.append(row['midprice'])
				idxs_buy.append((idx, row['midprice']))
				trades.put({'buy': True, 'time': idx, 'price': row['ask'], 'size': 0.01})
				last_buy = idx
				last_price = row['ask']
				available_btc += 0.01
				funds -= row['ask'] * 0.01
				buy_streak = False

	#return trades, zip(idxs_sell, ys_sell), buy_annotations, sell_annotations
	return trades, idxs_buy, idxs_sell, buy_annotations, sell_annotations

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

# for i in range(2,5):
# 	for j in range(2,5):
# 		for k in [datetime.timedelta(minutes=1), datetime.timedelta(minutes=5), datetime.timedelta(minutes=10), datetime.timedelta(minutes=20)]:
# 			for l in ['10min','30min','1h']:
# 				EPSILON_UPPER = i
# 				EPSILON_LOWER = j
# 				WAIT_TIME = k
# 				LOOKBACK = l
# 				print 'UPPER: {}\t LOWER: {}\t WAIT_TIME: {}\t LOOKBACK: {}'.format(EPSILON_UPPER, EPSILON_LOWER, str(WAIT_TIME), LOOKBACK)
# 				df = get_data()
# 				indicators_buy, indicators_sell = get_indicators(df)
# 				print 'Number of buy indicators: {}\t Number of sell indicators: {}'.format(len(indicators_buy), len(indicators_sell))
# 				trades, indicators_sell, buy_annotations, sell_annotations = run(df, indicators_buy, indicators_sell, funds=FUNDS)
# 				price_deltas = tools.compute_statistics(trades, funds=FUNDS)
# 				tools.compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=FUNDS)
# 				#graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations)
# print 'done'


def main():
	print 'UPPER: {}\t LOWER: {}\t WAIT_TIME: {}\t LOOKBACK: {}'.format(EPSILON_UPPER, EPSILON_LOWER, str(WAIT_TIME), LOOKBACK)
	df = get_data()
	indicators_buy, indicators_sell = get_indicators(df)
	print 'Number of buy indicators: {}\t Number of sell indicators: {}'.format(len(indicators_buy), len(indicators_sell))
	trades, indicators_buy, indicators_sell, buy_annotations, sell_annotations = run(df, indicators_buy, indicators_sell, funds=FUNDS)
	price_deltas = tools.compute_statistics(trades, funds=FUNDS)
	tools.compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=FUNDS)
	graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations)
	print 'done'

if __name__ == '__main__':
	main()

