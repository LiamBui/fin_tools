import numpy as np
import pandas as pd

def compute_returns(df):
	df['returns'] = (df['price'] - df.iloc[0]['price']) / df.iloc[0]['price']
	return df

def compute_statistics(q):
	pnl = 0.0
	available_btc = 0.0
	funds = 1000.0
	while not q.empty():
		try:
			trade = q.get()
			if trade['buy']:
				pnl -= trade['price'] * trade['size']
				available_btc += trade['size']
				funds -= trade['price'] * trade['size']
				print '{}: Bought {:.8f} BTC @ ${:.2f}.\t PnL: {:.2f}'.format(trade['time'], trade['size'], trade['price'], pnl)
			else:
				pnl += trade['price'] * trade['size']
				available_btc -= trade['size']
				funds += trade['price'] * trade['size']
				print '{}: Sold {:.8f} BTC @ ${:.2f}.\t Pnl: {:.2f}'.format(trade['time'], trade['size'], trade['price'], pnl)
		except:
			print 'Fuck u\n'
	print 'PnL: {:.2f}\t Funds: {:.2f}\t BTC: {:.8f}\n'.format(pnl, funds, available_btc)
	return q