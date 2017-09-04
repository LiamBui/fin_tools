import pandas as pd
import matplotlib.pyplot as plt
import tools, datetime, math, random
import Queue as queue
import numpy as np
import mpld3

VERBOSE = True

FUNDS = 100.0
UNITS = 60 # number of seconds
LOOKBACK = 20 * UNITS
BUY_SIZE = 0.01
SLIPPAGE_BUY = 0.01
SLIPPAGE_SELL = 0.01
HALFLIFE = 10 * UNITS
ALPHA = 1.0 - math.exp(math.log(0.5)/float(HALFLIFE))

LOCATION = 'C:/Users/lb/fin_tools/data/btc/20170511-20170521.log'


def funds_round(num, places=2, direction=math.floor):
    return direction(num * (10**places)) / float(10**places)

def update_avg_entry(avg_entry, available_btc, buy_price, buy_size):
	return (avg_entry * available_btc + (buy_size * buy_price)) / (available_btc + buy_size)

def update_pnl(pnl, avg_entry, sell_price, sell_size):
	return pnl + sell_size * (sell_price - avg_entry)

def sell(pnl, avg_entry, available_funds, available_btc, last_sell, sell_time, sell_price, sell_size):
	if random.random() < SLIPPAGE_SELL:

		pnl = update_pnl(pnl, avg_entry, sell_price, sell_size)
		available_btc = available_btc - sell_size
		available_funds = available_funds + sell_price * sell_size

		if VERBOSE:
			print '{}:\t Sold {:.2f} BTC @ {:.2f}, PnL: {:.2f}'.format(sell_time, sell_size, sell_price, pnl)

		return pnl, available_funds, available_btc, sell_time

	else:

		return pnl, available_funds, available_btc, last_sell



# ----------------------------------------------------------------------------



def buy(pnl, avg_entry, available_funds, available_btc, last_buy, buy_time, buy_price, buy_size):
	if random.random() < SLIPPAGE_BUY:

		avg_entry = update_avg_entry(avg_entry, available_btc, buy_price, buy_size)
		available_btc = available_btc + buy_size
		available_funds = available_funds - buy_price * buy_size

		if VERBOSE:
			print '{}:\t Bought {:.2f} BTC @ {:.2f}, Avg_entry: {:.2f}'.format(buy_time, buy_size, buy_price, avg_entry)

		return avg_entry, available_funds, available_btc, buy_time

	else:

		return avg_entry, available_funds, available_btc, last_buy




def get_data():

	df = pd.read_csv(LOCATION)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['datetime','bid_size','bid_price','mid_price','ask_price','ask_size','std','lower_band','upper_band','num']
	df = df.drop('lower_band', axis=1)
	df = df.drop('upper_band', axis=1)

	simple_rolling = df['mid_price'].rolling(LOOKBACK, min_periods=LOOKBACK)

	df['simple_ma'] = simple_rolling.mean()
	df['simple_std'] = simple_rolling.std()

	exponential_rolling = df['mid_price'].ewm(alpha=ALPHA, min_periods=LOOKBACK, adjust=False, ignore_na=True)

	df['exp_ma'] = exponential_rolling.mean()
	df['exp_std'] = exponential_rolling.std()

	df['lower_band_simple'] = df['simple_ma'] - 3.0 * df['simple_std'] 
	df['upper_band_simple'] = df['simple_ma'] + 3.0 * df['simple_std']

	df['lower_band_exp'] = df['exp_ma'] - 3.0 * df['exp_std']
	df['upper_band_exp'] = df['exp_ma'] + 3.0 * df['exp_std']

	return df.dropna()


def initialize_stats(df):
	# calculate mean
	n = len(df.index)
	avg = df['mid_price'].sum() / n

	# calculate std
	df['diff'] = df['mid_price'] - avg
	df['sq_diff'] = df['diff']**2
	ssd = df['sq_diff'].sum()
	var = ssd / (n-1)
	std = math.sqrt(var)

	return avg, var, std



# ----------------------------------------------------------------------------



def update_stats(avg, var, std, new, old, n):

	old_avg = avg
	new_avg = old_avg + (new - old) / n
	avg = new_avg
	var = var + (new - old) * (new - new_avg + old - old_avg) / (n - 1)
	if var > 0:
		std = math.sqrt(var)
	else:
		std = 0

	return avg, var, std




def graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations):

	xs = [x[0] for x in indicators_buy]
	ys = [x[1] for x in indicators_buy]

	xs1 = [x[0] for x in indicators_sell]
	ys1 = [x[1] for x in indicators_sell]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['mid_price'], color='black', label='MidPrice')
	buy = ax1.scatter(xs, ys, color='green', label='Buy indicators')
	sell = ax1.scatter(xs1, ys1, color='red', label='Sell indicators')

	# SIMPLE	
	ax1.plot(df['datetime'], df['upper_band_simple'], color='blue', label='Upper')
	ax1.plot(df['datetime'], df['lower_band_simple'], color='blue', label='Lower')

	# EXP
	ax1.plot(df['datetime'], df['upper_band_exp'], color='orange', label='Upper EXP')
	ax1.plot(df['datetime'], df['lower_band_exp'], color='orange', label='Lower EXP')

	tooltip = mpld3.plugins.PointLabelTooltip(buy, labels=buy_annotations)
	mpld3.plugins.connect(fig, tooltip)
	tooltip2 = mpld3.plugins.PointLabelTooltip(sell, labels=sell_annotations)
	mpld3.plugins.connect(fig, tooltip2)

	# plt.legend(loc='upper left')
	# plt.ylabel('Price (USD)')
	# plt.xlabel('Time')
	# plt.show()

	mpld3.show()

	return


def graph_simple(df):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['mid_price'], color='black', label='MidPrice')

	# SIMPLE	
	ax1.plot(df['datetime'], df['simple_ma'], color='blue', label='Simple MA')
	# EXP
	ax1.plot(df['datetime'], df['exp_ma'], color='orange', label='Exp MA')

	plt.legend(loc='upper left')
	plt.ylabel('Price (USD)')
	plt.xlabel('Time')
	plt.show()

	return

def run_value(df, available_funds = FUNDS):

	if VERBOSE:
		print 'Running value strategy.'


	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []
	pnls = []


	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	last_midprice = df.loc[(df.index < datetime.datetime(2017,5,11,5,0,0)) & (df.index > datetime.datetime(2017,5,11,4,59,58))]

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		if row['num'] < 60:
			continue

		if idx > last_midprice.iloc[0].name + datetime.timedelta(days=1) and idx < last_midprice.iloc[0].name + datetime.timedelta(days=1, hours=1):
			# print idx
			# print last_midprice.iloc[0]['mid_price']
			# print row['mid_price']
			if row['mid_price'] < 0.96 * last_midprice.iloc[0]['mid_price']:
				price_buy = funds_round(row['mid_price'], direction=math.floor)
				funds_required = price_buy * BUY_SIZE
				if available_funds > funds_required and row['exp_ma'] > row['simple_ma']:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, available_funds / price_buy)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
						last_midprice = df.loc[df.index == idx]

		if last_midprice.iloc[0].name < idx - datetime.timedelta(days=1, hours=1):
			last_midprice = df.loc[(df.index > last_midprice.iloc[0].name + datetime.timedelta(days=1)) & (df.index < last_midprice.iloc[0].name + datetime.timedelta(days=1, hours=1))]


		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:
			price_sell = funds_round(row['mid_price'], direction=math.ceil)
			if available_btc > 0 and row['exp_ma'] < row['simple_ma']:
				old_pnl = pnl
				pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
				if old_pnl != pnl:
					indicators_sell.append((idx, row['mid_price']))
					sell_annotations.append(row['mid_price'])
					pnls.append(pnl)

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations, pnls

def main():

	df = get_data()

	returns = []

	pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations, pnls = run_value(df, available_funds=FUNDS)
	returns.append(pnl)

	print 'Pnl:\t {}'.format(pnl)

	wins = [pnls[x+1] - pnls[x] for x in range(len(pnls)-1) if pnls[x+1] - pnls[x] > 0]
	losses = [pnls[x+1] - pnls[x] for  x in range(len(pnls)-1) if pnls[x+1] - pnls[x]  < 0]

	prob_win = len(wins) / float(len(wins) + len(losses))
	prob_lose = len(losses) / float(len(wins) + len(losses))
	print 'Percent wins:\t {}'.format(prob_win)
	print 'Percent losses:\t {}'.format(prob_lose)

	avg_win = sum(wins) / float(len(wins))
	avg_loss = sum(losses) / float(len(losses))

	print 'Average win:\t {}'.format(avg_win)
	print 'Average loss:\t {}'.format(avg_loss)
	print 'Expected value:\t {}'.format(avg_win * prob_win + avg_loss * prob_lose)

	avg = np.mean(returns)
	std = np.std(returns)
	min_return = np.min(returns)
	max_return = np.max(returns)

	print 'Average return: {:.2f}\t Std: {:.2f}\t Min: {:.2f}\t Max: {:.2f}\n'.format(avg, std, min_return, max_return)

if __name__ == '__main__':
	main()