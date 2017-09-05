import math
import pandas as pd

def funds_round(num, places=2, direction=math.floor):
    return direction(num * (10**places)) / float(10**places)

def update_avg_entry(avg_entry, available_btc, buy_price, buy_size):
	return (avg_entry * available_btc + (buy_size * buy_price)) / (available_btc + buy_size)

def update_pnl(pnl, avg_entry, sell_price, sell_size):
	return pnl + sell_size * (sell_price - avg_entry)

def get_data(loc):

	df = pd.read_csv(loc)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['datetime','bid_size','bid_price','mid_price','ask_price','ask_size','std','lower_band','upper_band','num']
	df = df.drop('lower_band', axis=1)
	df = df.drop('upper_band', axis=1)

	return df.dropna()
