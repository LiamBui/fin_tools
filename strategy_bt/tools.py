import numpy as np
import pandas as pd

def compute_returns(df):
	df['returns'] = (df['price'] - df.iloc[0]['price']) / df.iloc[0]['price']
	return df

def compute_statistics(q, funds=1000.0):
	pnl = 0.0
	available_btc = 0.0
	average_entry = 0.0
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
				available_btc -= trade['size']
				funds += trade['price'] * trade['size']
				print '{}: Sold {:.8f} BTC @ ${:.2f}.\t Pnl: {:.2f}'.format(trade['time'], trade['size'], trade['price'], pnl)
		except:
			print 'Fuck u\n'
	print 'PnL: {:.2f}\t Funds: {:.2f}\t BTC: {:.8f}\n'.format(pnl, funds, available_btc)
	return q