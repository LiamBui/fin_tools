import pandas as pd
import matplotlib.pyplot as plt
import queue, tools, datetime

def get_data():
	location = '../data/output.log'
	df = pd.read_csv(location)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	#df = df.drop(df.columns[[0]], axis=1)
	df.columns = ['datetime','tilt','tilt_up','price_delta','bid','ask','midprice']#,'bid_size','ask_size']
	#df.index.name = 'datetime'

	return df

def graph(df, indicators_buy, indicators_sell):

	xs = [x[0] for x in indicators_buy]
	ys = [x[1] for x in indicators_buy]

	xs1 = [x[0] for x in indicators_sell]
	ys1 = [x[1] for x in indicators_sell]

	def onpick3(event):
		print type(event)
		ind = event.ind
		print event

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['midprice'], color='black', label='MidPrice')
	ax1.scatter(xs, ys, color='green', label='Buy indicators')
	ax1.scatter(xs1, ys1, color='red', label='Sell indicators')
	fig.canvas.mpl_connect('pick_event', onpick3)

	plt.legend(loc='upper left')
	plt.ylabel('Price (USD)')
	plt.xlabel('Time')
	plt.show()

def run(df, indicators_buy, indicators_sell, funds=1000.0):
	(idxs_buy, _) = zip(*indicators_buy)
	(idxs_sell, _) = zip(*indicators_sell)
	print len(idxs_buy), len(idxs_sell)
	available_btc = 0.0 
	pnl = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask']

	for idx, row in df.iterrows():
		if funds > row['ask'] * 0.01:
			if idx > last_buy + datetime.timedelta(minutes=15):
				if idx in idxs_buy:
					trades.put({'buy': True, 'time': idx, 'price': row['ask'], 'size': 0.01})
					last_buy = idx
					last_price = row['ask']
					available_btc += 0.01
					funds -= row['ask'] * 0.01
					pnl -= row['ask'] * 0.01
		if available_btc > 0.0:
			if idx in idxs_sell or (row['bid'] <= last_price - 1.50 and row['tilt'] <= 0.3):
				trades.put({'buy': False, 'time': idx, 'price': row['bid'], 'size': 0.01})
				available_btc -= 0.01
				funds += row['bid'] * 0.01
				pnl += row['bid'] * 0.01

	print 'PnL : ${:.2f}\t Funds : ${:.2f}\t Available BTC : {:.8f}'.format(pnl, funds, available_btc)

	return trades

def get_indicators(df, buy_ratio, buy_tilt, sell_ratio, sell_tilt):
	indicators_buy = []
	indicators_sell = []

	df = df.sort_index()
	rolling = df['tilt_up'].rolling('10min', min_periods=1)
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
	indicators_buy, indicators_sell = get_indicators(df, 0.5, 0.95, 0.3, 0.05)
	trades = run(df, indicators_buy, indicators_sell, funds=100.0)
	tools.compute_statistics(trades, funds=100.0)
	graph(df, indicators_buy, indicators_sell)

if __name__ == '__main__':
	main()