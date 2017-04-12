import numpy as np
import pandas as pd
import math

def compute_returns(df):
	df['returns'] = (df['price'] - df.iloc[0]['price']) / df.iloc[0]['price']
	return df

def compute_statistics(q, funds=1000.0):
	pnl = 0.0
	available_btc = 0.0
	average_entry = 0.0
	price_deltas = []
	while not q.empty():
		try:
			trade = q.get()
			
			if trade['buy']:
				average_entry = (average_entry * available_btc + trade['size'] * trade['price']) / (available_btc + trade['size'])
				available_btc += trade['size']
				funds -= trade['price'] * trade['size']
				print '{}: Bought {:.8f} BTC @ ${:.2f}.\t PnL: {:.2f}'.format(trade['time'], trade['size'], trade['price'], pnl)
			else:
				pnl = pnl + (trade['price'] * trade['size'] - average_entry * trade['size'])
				price_deltas.append(trade['price'] * trade['size'] - average_entry * trade['size'])
				available_btc -= trade['size']
				funds += trade['price'] * trade['size']
				print '{}: Sold {:.8f} BTC @ ${:.2f}.\t Pnl: {:.2f}'.format(trade['time'], trade['size'], trade['price'], pnl)
		except:
			print 'Fuck u\n'
	# print 'PnL: {:.2f}\t Funds: {:.2f}\t BTC: {:.8f}\n'.format(pnl, funds, available_btc)
	return price_deltas

def compute_returns_sharpe_maxdrawdown(price_deltas, df, funds=1000.0):
	pnl = 0.0
	available_btc = 0.0
	average_entry = 0.0
	max_drawdown = 0

	returns = []
	drawdowns = []
	returns.append((funds + price_deltas[0]) / funds)
	for x in price_deltas[1:]:
		returns.append(returns[-1] + x / funds)
	for x in range(len(returns)-1):
		drawdowns.append(returns[x+1] - returns[x])
	max_drawdown = np.min([0,np.min(drawdowns)])

	rf_return = 1 + ((df.iloc[len(df.index) - 1]['midprice'] - df.iloc[0]['midprice']) / df.iloc[0]['midprice']) * (funds / df.iloc[0]['midprice'])
	std = np.std(returns)

	sharpe = (returns[-1] - rf_return) / std
	sharpe = sharpe * math.sqrt(365*24) #annualized by hour

	print 'Returns: {:.4f}\t Sharpe Ratio: {:.4f}\t Max Drawdown: {:.4f}\n'.format(returns[-1], sharpe, max_drawdown)

	return

def compute_returns_benchmark_maxdrawdown(price_deltas, df, funds=1000.0):
	pnl = 0.0
	available_btc = 0.0
	average_entry = 0.0
	max_drawdown = 0

	returns = []
	drawdowns = []
	returns.append((funds + price_deltas[0]) / funds)
	for x in price_deltas[1:]:
		returns.append(returns[-1] + x / funds)
	for x in range(len(returns)-1):
		drawdowns.append(returns[x+1] - returns[x])
	drawdowns.append(20000)
	max_drawdown = np.min([0,np.min(drawdowns)])

	rf_return = 1 + ((df.iloc[len(df.index) - 1]['midprice'] - df.iloc[0]['midprice']) / df.iloc[0]['midprice']) * (funds / df.iloc[0]['midprice'])
	#very approximate risk-free return

	print 'Returns: {:.4f}\t Benchmark: {:.4f}\t Max Drawdown: {:.4f}\n'.format(returns[-1], rf_return, max_drawdown)

	return