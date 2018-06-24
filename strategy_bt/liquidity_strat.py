import numpy as np
import pandas as pd
import preprocess, queue
from scipy import stats

def run(df1, df2):
	# merged, keeping only last of non-unique indices
	df1 = df1[~df1.index.duplicated(keep='last')]
	df2 = df2[~df2.index.duplicated(keep='last')]
	df3 = df1.join(df2.reindex(df1.index, method='nearest'), lsuffix='_coinbase', rsuffix='_bitfinex')

	df3['returns_coinbase'] = (df3['price_coinbase'] - df3.iloc[0]['price_coinbase']) / df3.iloc[0]['price_coinbase']
	df3['returns_bitfinex'] = (df3['price_bitfinex'] - df3.iloc[0]['price_bitfinex']) / df3.iloc[0]['price_bitfinex']

	# m, b, corr_coef, p_value, std_err = stats.linregress(df['returns_coinbase'], df['returns_bitfinex'])

	# print 'Slope: {:.2f} \t Intercept: {:.2f} \t R: {:.2f} \t P-value: {:.2f} \t'.format(m, b, corr_coef, p_value)
	# print len(df.index)

	df1_rolling = df3['price_coinbase'].rolling('3min', min_periods=1)
	df2_rolling = df3['price_bitfinex'].rolling('3min', min_periods=1)

	df3['priceAvg_coinbase'] = df1_rolling.mean()
	df3['priceAvg_bitfinex'] = df2_rolling.mean()
	df3['stdAvg_coinbase'] = df1_rolling.std()
	df3['stdAvg_bitfinex'] = df2_rolling.std()
	df3['indicator_coinbase'] = df3['price_coinbase'] > df3['priceAvg_coinbase'] + 1.0*df3['stdAvg_coinbase']
	df3['indicator_bitfinex'] = df3['price_bitfinex'] > df3['priceAvg_bitfinex'] + 5.0*df3['stdAvg_bitfinex']

	df, test = preprocess.generate_train_test_split(df3, 133333, train_size=1.0)

	funds = 1000.0
	available_btc = 0.0
	trades = queue.Queue()
	curr = []
	pnl = 0.0

	for idx, row in df.iterrows():
		if funds > 0.0: # if we have money
			if row['indicator_bitfinex'] == True and row['indicator_coinbase'] == False: # buy signal
				if row['price_coinbase'] * row['size_coinbase'] > funds: # big size
					trades.put({'buy': True, 'time': idx, 'price': row['price_coinbase'], 'size': funds / row['price_coinbase']})
					curr.append((row, funds/row['price_coinbase'], 'buy'))
					# print 'Bought {:.8f} BTC @ ${:.2f}'.format(funds / row['price_coinbase'], row['price_coinbase'])
					pnl -= funds
					funds = 0.0
					available_btc += funds / row['price_coinbase']
				else: # small size
					trades.put({'buy': True, 'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']})
					curr.append((row, row['size_coinbase'], 'buy'))
					# print 'Bought {:.8f} BTC @ ${:.2f}'.format(row['size_coinbase'], row['price_coinbase'])
					pnl -= row['price_coinbase'] * row['size_coinbase']
					funds -= row['price_coinbase'] * row['size_coinbase']
					available_btc += row['size_coinbase']
			elif curr and (row['price_coinbase'] < curr[-1][0]['priceAvg_coinbase'] - 3 * row['stdAvg_coinbase']): 
				if available_btc > 0.0:
					if row['size_coinbase'] >= available_btc:
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': available_btc})
						curr.append((row, available_btc, 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(available_btc, row['price_coinbase'])
						pnl += row['price_coinbase'] * available_btc
						funds += row['price_coinbase'] * available_btc
						available_btc = 0.0
					else: # ignores stopper
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']})
						curr.append((row, row['size_coinbase'], 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(row['size_coinbase'], row['price_coinbase'])
						pnl += row['price_coinbase'] * row['size_coinbase']
						funds += row['price_coinbase'] * row['size_coinbase']
						available_btc -= row['size_coinbase']
			elif (row['price_bitfinex'] < row['priceAvg_bitfinex'] + 3 * row['stdAvg_bitfinex']) and row['indicator_coinbase'] == False: # still on the up, but signal falling
				if available_btc > 0.0:
					if row['size_coinbase'] >= available_btc:
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': available_btc})
						curr.append((row, available_btc, 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(available_btc, row['price_coinbase'])
						pnl += row['price_coinbase'] * available_btc
						funds += row['price_coinbase'] * available_btc
						available_btc = 0.0
					else: # ignores stopper
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']})
						curr.append((row, row['size_coinbase'], 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(row['size_coinbase'], row['price_coinbase'])
						pnl += row['price_coinbase'] * row['size_coinbase']
						funds += row['price_coinbase'] * row['size_coinbase']
						available_btc -= row['size_coinbase']
			else:
				continue
		else:
			if curr and (row['price_coinbase'] < curr[-1][0]['priceAvg_coinbase'] - 3 * row['stdAvg_coinbase']): 
				if available_btc > 0.0:
					if row['size_coinbase'] >= available_btc:
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': available_btc})
						curr.append((row, available_btc, 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(available_btc, row['price_coinbase'])
						pnl += row['price_coinbase'] * available_btc
						funds += row['price_coinbase'] * available_btc
						available_btc = 0.0
					else: # ignores stopper
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']})
						curr.append((row, row['size_coinbase'], 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(row['size_coinbase'], row['price_coinbase'])
						pnl += row['price_coinbase'] * row['size_coinbase']
						funds += row['price_coinbase'] * row['size_coinbase']
						available_btc -= row['size_coinbase']
			elif (row['price_bitfinex'] < row['priceAvg_bitfinex'] + 3 * row['stdAvg_bitfinex']) and row['indicator_coinbase'] == False: # still on the up, but signal falling
				if available_btc > 0.0:
					if row['size_coinbase'] >= available_btc:
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': available_btc})
						curr.append((row, available_btc, 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(available_btc, row['price_coinbase'])
						pnl += row['price_coinbase'] * available_btc
						funds += row['price_coinbase'] * available_btc
						available_btc = 0.0
					else: # ignores stopper
						trades.put({'buy': False, 'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']})
						curr.append((row, row['size_coinbase'], 'sell'))
						# print 'Sold {:.8f} BTC @ ${:.2f}'.format(row['size_coinbase'], row['price_coinbase'])
						pnl += row['price_coinbase'] * row['size_coinbase']
						funds += row['price_coinbase'] * row['size_coinbase']
						available_btc -= row['size_coinbase']
			else:
				continue




		# 		current_trade = {'time': idx, 'price': row['price_coinbase'], 'size': row['size_coinbase']}
		# 		trades.append(row)
		# 		pnl -= row['price_coinbase'] * row['size_coinbase']
		# else:
		# 	if row['price_coinbase'] > current_trade['price'] + row['stdAvg_coinbase']:
		# 		if row['size_coinbase'] >= current_trade['size']:
		# 			pnl += row['price_coinbase'] * current_trade['size']
		# 			current_trade = {}
		# 			trades = []
		# 		else:
		# 			pnl += row['price_coinbase'] * row['size_coinbase']
		# 			current_trade['size'] -= row['size_coinbase']
		# 	else:
		# 		if row['price_coinbase'] < current_trade['price'] - row['stdAvg_coinbase']:
		# 			if row['size_coinbase'] >= current_trade['size']:
		# 				pnl += row['price_coinbase'] * current_trade['size']
		# 				current_trade = {}
		# 				trades = []
		# 			else:
		# 				pnl += row['price_coinbase'] * row['size_coinbase']
		# 				current_trade['size'] -= row['size_coinbase']

	# if len(trades) > 0:
	# 	pnl += df.tail(n=5)['price_coinbase'] * current_trade['size']
	# trades = []
	# current_trade = {}

	if available_btc > 0:
		most_recent = df.tail(n=1).iloc[0]
		trades.put({'buy': False, 'time': most_recent.index.name, 'price': most_recent['price_coinbase'], 'size': available_btc})
		pnl += most_recent['price_coinbase'] * available_btc
		funds += most_recent['price_coinbase'] * available_btc
		available_btc = 0.0

	print 'PnL : ${:.2f}\t Funds : ${:.2f}\t Available BTC : {:.8f}'.format(pnl, funds, available_btc)
	print trades.qsize()

	print df.head(n=5)
	print df.tail(n=5)

	return trades
