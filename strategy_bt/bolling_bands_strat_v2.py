import pandas as pd
import matplotlib.pyplot as plt
import tools, datetime, math, random
import Queue as queue
import numpy as np
import mpld3

VERBOSE = False

FUNDS = 100.0
UNITS = 60 # number of seconds
LOOKBACK = 5 * UNITS
EPSILON_UPPER = 1.0
EPSILON_LOWER = 0.6
WAIT_SELL = datetime.timedelta(minutes=20)
WAIT_BUY = datetime.timedelta(minutes=5)
WAIT_PUKE = datetime.timedelta(minutes=1)
WAIT_PUKE2 = datetime.timedelta(minutes=90)
WAIT_NEXT_BUY = datetime.timedelta(seconds=60)
HALFLIFE = 10 * UNITS
HALFLIFE2 = 5 * UNITS
ALPHA = 1.0 - math.exp(math.log(0.5)/float(HALFLIFE))
ALPHA2 = 1.0 - math.exp(math.log(0.5)/float(HALFLIFE2))
BUY_SIZE = 0.01
SLIPPAGE_BUY = 1
SLIPPAGE_SELL = 1
EPSILON_UPPER2 = 4.0

#LOCATIONS = ['C:/Users/lb/fin_tools/data/btc/20170511-20170521.log','C:/Users/lb/fin_tools/data/btc/20170721-20170729.log']
LOCATIONS = ['C:/Users/lb/fin_tools/data/btc/20170730-20170807.log','C:/Users/lb/fin_tools/data/btc/20170808-20170815.log',
             'C:/Users/lb/fin_tools/data/btc/20170816-20170823.log','C:/Users/lb/fin_tools/data/btc/20170824-20170901.log']

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



# ----------------------------------------------------------------------------



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



# ----------------------------------------------------------------------------



def get_data(loc):

	df = pd.read_csv(loc)

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



# ----------------------------------------------------------------------------



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
	# ax1.plot(df['datetime'], df['upper_band_simple'], color='blue', label='Upper')
	# ax1.plot(df['datetime'], df['lower_band_simple'], color='blue', label='Lower')

	# EXP
	# ax1.plot(df['datetime'], df['upper_band_exp'], color='orange', label='Upper EXP')
	# ax1.plot(df['datetime'], df['lower_band_exp'], color='orange', label='Lower EXP')

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


# ----------------------------------------------------------------------------


def graph_simple(df):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['mid_price'], color='black', label='MidPrice')
	plt.legend(loc='upper left')
	plt.ylabel('Price (USD)')
	plt.xlabel('Time')
	plt.show()

	return

# ----------------------------------------------------------------------------


def run_simple(df, available_funds=1000.00):

	if VERBOSE:
		print 'Running simple Bollinger band strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df_init = df.head(n=LOOKBACK)
	df = df.tail(n=(len(df.index)-LOOKBACK))

	avg, var, std = initialize_stats(df_init)

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		avg, var, std = update_stats(avg, var, std, row['mid_price'], df.iloc[df.index.get_loc(idx)-LOOKBACK]['mid_price'], LOOKBACK)

		sell_stop = 2.0 * std
		sell_point = 1.0 * std
		upper_band = avg + EPSILON_UPPER * std
		lower_band = avg - EPSILON_LOWER * std

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# sell 
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] > upper_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / HALFLIFE
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue


			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] < lower_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
	
	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations


# ----------------------------------------------------------------------------


def run_exp(df, available_funds=1000.0):

	if VERBOSE:
		print 'Running exponential Bollinger Bands strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = 2.0 * row['exp_std']
		sell_point = 1.0 * row['exp_std']
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# sell 
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] > upper_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / HALFLIFE
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue


			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] < lower_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations

# ----------------------------------------------------------------------------

def initialize_stats_exp(df):
	rolling = df['mid_price'].ewm(alpha=ALPHA, min_periods=LOOKBACK)
	mean = rolling.mean()
	var = rolling.var()
	std = rolling.std()
	return mean.tail(n=1).iloc[0], var.tail(n=1).iloc[0], std.tail(n=1).iloc[0]

def update_stats_exp(new, mean, var):
	diff = new - mean
	incr = ALPHA * diff
	mean = mean + incr
	var = (1.0 - ALPHA) * (var + diff * incr)
	std = math.sqrt(var)
	return mean, var, std

def run_exp_online(df, available_funds=1000.00):

	if VERBOSE:
		print 'Running online exponential strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df_init = df.head(n=LOOKBACK)
	df = df.tail(n=(len(df.index)-LOOKBACK))

	avg, var, std = initialize_stats_exp(df_init)

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		avg, var, std = update_stats_exp(row['mid_price'], avg, var)

		sell_stop = 2.0 * std
		sell_point = 1.2 * std
		upper_band = avg + EPSILON_UPPER * std
		lower_band = avg - EPSILON_LOWER * std

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# sell 
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] > upper_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = buy(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / float(HALFLIFE)
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue


			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] < lower_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
	
	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations


def run_reverse_exp_online(df, available_funds=1000.00):

	if VERBOSE:
		print 'Running reverse online exponential strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df_init = df.head(n=LOOKBACK)
	df = df.tail(n=(len(df.index)-LOOKBACK))

	avg, var, std = initialize_stats_exp(df_init)

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		avg, var, std = update_stats_exp(row['mid_price'], avg, var)

		sell_stop = 2.0 * std
		sell_point = 1.2 * std
		upper_band = avg + EPSILON_UPPER * std
		lower_band = avg - EPSILON_LOWER * std

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			# if idx < datetime.datetime(2017,05,11,11,33) and idx > datetime.datetime(2017,05,11,11,32):
			# 	print 'idx: {}\tavg: {:.2f}\tstd: {:.2f}'.format(idx, avg, std)
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
						# print 'Bought. Std: {:.2f}'.format(std)

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / float(HALFLIFE)
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations

# ----------------------------------------------------------------------------

def run_reverse_exp(df, available_funds=1000.0):

	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = 2.0 * row['exp_std']
		sell_point = 1.0 * row['exp_std']
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			# if idx < datetime.datetime(2017,05,11,11,33) and idx > datetime.datetime(2017,05,11,11,32):
			# 	print 'idx: {}\tavg: {:.2f}\tstd: {:.2f}'.format(idx, row['exp_ma'], row['exp_std'])
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, available_funds / price_buy)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
						# print 'Bought. Std: {:.2f}'.format(row['exp_std'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / HALFLIFE
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations


# ----------------------------------------------------------------------------

def run_reverse_exp_sellpoint(df, sell_epsilon, available_funds=1000.0):

	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy w/o sell point.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = 2.0 * row['exp_std']
		sell_point = sell_epsilon * row['exp_std']
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / float(HALFLIFE) 
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					# elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
					# 	pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					# 	if old_pnl != pnl:
					# 		indicators_sell.append((idx, row['mid_price']))
					# 		sell_annotations.append(row['mid_price'])
					# 	continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations


# ----------------------------------------------------------------------------

def run_reverse_exp_sellpoint_static(df, sell_epsilon, available_funds=1000.0):

	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy w/o sell point.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = 2.0 * row['exp_std']
		sell_point = sell_epsilon
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / float(HALFLIFE) 
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					# elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
					# 	pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					# 	if old_pnl != pnl:
					# 		indicators_sell.append((idx, row['mid_price']))
					# 		sell_annotations.append(row['mid_price'])
					# 	continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations

# ----------------------------------------------------------------------------


def run_reverse_exp_no_sellpoint(df, available_funds=1000.0):

	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy w/o sell point.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, BUY_SIZE)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations


# ----------------------------------------------------------------------------

def run_reverse_exp_stopper(df, stopper_epsilon, available_funds=1000.0):

	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy.'

	indicators_buy = []
	indicators_sell = []
	buy_annotations = []
	sell_annotations = []

	available_btc = 0.0 
	pnl = 0.0
	avg_entry = 0.0
	trades = queue.Queue()
	last_buy = df.iloc[0]['datetime']
	last_sell = df.iloc[0]['datetime']
	last_price = df.iloc[0]['ask_price']

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = stopper_epsilon * row['exp_std']
		sell_point = 1.0 * row['exp_std']
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			# if idx < datetime.datetime(2017,05,11,11,33) and idx > datetime.datetime(2017,05,11,11,32):
			# 	print 'idx: {}\tavg: {:.2f}\tstd: {:.2f}'.format(idx, row['exp_ma'], row['exp_std'])
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, available_funds / price_buy)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
						# print 'Bought. Std: {:.2f}'.format(row['exp_std'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / HALFLIFE
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
						continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations

# ----------------------------------------------------------------------------

def run_reverse_exp_v2(df, available_funds=1000.00):
	if VERBOSE:
		print 'Running reverse exponential Bollinger Bands strategy.'

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

	df = df.tail(n=(len(df.index)-LOOKBACK))

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		sell_stop = 3.0 * row['exp_std']
		sell_point = 3.0 * row['exp_std']
		upper_band = row['exp_ma'] + EPSILON_UPPER * row['exp_std']
		lower_band = row['exp_ma'] - EPSILON_LOWER * row['exp_std']

		if row['num'] < 60:
			continue

		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			# buy
			# if idx < datetime.datetime(2017,05,11,11,33) and idx > datetime.datetime(2017,05,11,11,32):
			# 	print 'idx: {}\tavg: {:.2f}\tstd: {:.2f}'.format(idx, row['exp_ma'], row['exp_std'])
			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_sell + WAIT_BUY and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, available_funds / price_buy)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])
						# print 'Bought. Std: {:.2f}'.format(row['exp_std'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL and avg_entry > 0:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
						pnls.append(pnl)
					continue

				elif avg_entry > 0:
					old_pnl = pnl
					exp = pd.Timedelta(idx - last_buy).seconds / HALFLIFE
					if exp > 1 and row['bid_price'] > (avg_entry + sell_point / (2**int(exp))):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
							pnls.append(pnl)
						continue
					elif exp <= 1 and row['bid_price'] > (avg_entry + sell_point):
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
							pnls.append(pnl)
						continue
					elif row['bid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
						pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
						if old_pnl != pnl:
							indicators_sell.append((idx, row['mid_price']))
							sell_annotations.append(row['mid_price'])
							pnls.append(pnl)
						continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations, pnls


# ----------------------------------------------------------------------------
def initialize_stats_exp2(df):
	rolling = df['mid_price'].ewm(alpha=ALPHA2, min_periods=LOOKBACK)
	mean = rolling.mean()
	var = rolling.var()
	std = rolling.std()
	return mean.tail(n=1).iloc[0], var.tail(n=1).iloc[0], std.tail(n=1).iloc[0]

def update_stats_exp2(new, mean, var):
	diff = new - mean
	incr = ALPHA2 * diff
	mean = mean + incr
	var = (1.0 - ALPHA2) * (var + diff * incr)
	std = math.sqrt(var)
	return mean, var, std

def run_reverse_exp_online_v2(df, available_funds=1000.00):

	if VERBOSE:
		print 'Running reverse online exponential strategy.'

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

	df_init = df.head(n=LOOKBACK)
	df = df.tail(n=(len(df.index)-LOOKBACK))

	avg, var, std = initialize_stats_exp(df_init)
	avg2, var2, std2 = initialize_stats_exp2(df_init)
	missed = False
	miss_idx = df.iloc[0]['datetime']

	for idx, row in df.iloc[LOOKBACK:].iterrows():

		avg, var, std = update_stats_exp(row['mid_price'], avg, var)
		avg2, var2, std2 = update_stats_exp2(row['mid_price'], avg2, var2)

		sell_stop = 2 * std
		upper_band = avg + EPSILON_UPPER * std
		upper_band2 = avg + EPSILON_UPPER2 * std
		lower_band = avg - EPSILON_LOWER * std

		if row['num'] < 60:
			continue


		if row['mid_price'] != df.iloc[df.index.get_loc(idx)-1]['mid_price']:

			price_buy = funds_round(row['mid_price'], direction=math.floor)
			funds_required = price_buy * BUY_SIZE
			if available_funds > funds_required:
				if row['mid_price'] > upper_band and idx > last_buy + WAIT_NEXT_BUY:
					old_ae = avg_entry
					avg_entry, available_funds, available_btc, last_buy = buy(pnl, avg_entry, available_funds, available_btc, last_buy, idx, price_buy, available_funds / price_buy)
					if old_ae != avg_entry:
						indicators_buy.append((idx, row['mid_price']))
						buy_annotations.append(row['mid_price'])

			# sell
			if available_btc > 0:
				price_sell = funds_round(row['mid_price'], direction=math.ceil)
				if row['mid_price'] < lower_band and idx > last_buy + WAIT_SELL:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
						pnls.append(pnl)
					continue
				elif row['mid_price'] < (avg_entry - sell_stop) and idx > last_buy + WAIT_PUKE:
					old_pnl = pnl
					pnl, available_funds, available_btc, last_sell = sell(pnl, avg_entry, available_funds, available_btc, last_sell, idx, price_sell, available_btc)
					if old_pnl != pnl:
						indicators_sell.append((idx, row['mid_price']))
						sell_annotations.append(row['mid_price'])
						pnls.append(pnl)
					continue

	if VERBOSE:
		print 'PnL: {:.2f}'.format(pnl)

	return pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations, pnls

# ----------------------------------------------------------------------------


# def main():
# 	df = get_data()

# 	returns = []
# 	# pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations = run_simple(df, available_funds=FUNDS)

# 	pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations = run_reverse_exp(df, available_funds=FUNDS)
# 	returns.append(pnl)

# 	avg = np.mean(returns)
# 	std = np.std(returns)
# 	min_return = np.min(returns)
# 	max_return = np.max(returns)
# 	print LOCATION
# 	print 'Average return: {:.2f}\t Std: {:.2f}\t Min: {:.2f}\t Max: {:.2f}'.format(avg, std, min_return, max_return)
# 	# graph_simple(df)
# 	# print 'buy indicators: {}, buy annotations: {}'.format(len(indicators_buy), len(buy_annotations))
# 	# print 'sell indicators: {}, sell annotations: {}'.format(len(indicators_sell), len(sell_annotations))
# 	# graph(df, indicators_buy, indicators_sell, buy_annotations, sell_annotations)

# 	print 'Done'

# if __name__ == '__main__':
# 	main()

# ----------------------------------------------------------------------------


overall_days = 0
overall_returns = []
overall_prob_win = []
overall_prob_loss = []
overall_avg_win = []
overall_avg_loss = []
overall_ev = []
overall_len = []

for i in range(3):
	total_days = 0
	total_returns = 0
	total_wins = []
	total_losses = []
	total_len = 0

	for location in LOCATIONS:
		df = get_data(location)

		# find number of days
		num_days = df.iloc[len(df.index) - 1].name - df.iloc[0].name
		total_days += num_days.days

		returns = []

		pnl, indicators_buy, indicators_sell, buy_annotations, sell_annotations, pnls = run_reverse_exp_online_v2(df, available_funds=FUNDS)
		pnls.insert(0, 0)
		returns.append(pnl)

		total_returns += pnl
		total_len += len(pnls)

		wins = [pnls[x+1] - pnls[x] for x in range(len(pnls)-1) if pnls[x+1] - pnls[x] > 0]
		losses = [pnls[x+1] - pnls[x] for  x in range(len(pnls)-1) if pnls[x+1] - pnls[x]  < 0]

		total_wins = total_wins + wins
		total_losses = total_losses + losses

		try:
			prob_win = len(wins) / float(len(wins) + len(losses))
			prob_lose = len(losses) / float(len(wins) + len(losses))
		except:
			prob_win = 0
			prob_lose = 0

		try:
			avg_win = sum(wins) / float(len(wins))
		except:
			avg_win = 0
		try:
			avg_loss = sum(losses) / float(len(losses))
		except:
			avg_loss = 0

		avg = np.mean(returns)
		std = np.std(returns)
		min_return = np.min(returns)
		max_return = np.max(returns)

		if VERBOSE:
			print 'Location: {}'.format(location)
			# print 'Percent wins:\t {}'.format(prob_win)
			# print 'Percent losses:\t {}'.format(prob_lose)
			# print 'Average win:\t {}'.format(avg_win)
			# print 'Average loss:\t {}'.format(avg_loss)
			# print 'Expected value:\t {}'.format(avg_win * prob_win + avg_loss * prob_lose)
			print 'Average return: {:.2f}\t Std: {:.2f}\t Min: {:.2f}\t Max: {:.2f}\n'.format(avg, std, min_return, max_return)

		# print 'Location: {}'.format(location)
		# print 'Average return: {:.2f}\t Std: {:.2f}\t Min: {:.2f}\t Max: {:.2f}'.format(avg, std, min_return, max_return)

	total_days = int(total_days)
	total_prob_win = len(total_wins) / float(len(total_wins) + len(total_losses))
	total_prob_loss = len(total_losses) / float(len(total_wins) + len(total_losses))
	total_avg_win = sum(total_wins) / float(len(total_wins))
	total_avg_loss = sum(total_losses) / float(len(total_losses))
	total_ev = total_avg_win * total_prob_win + total_avg_loss * total_prob_loss

	overall_days = total_days
	overall_returns.append(total_returns)
	overall_prob_win.append(total_prob_win)
	overall_prob_loss.append(total_prob_loss)
	overall_avg_win.append(total_avg_win)
	overall_avg_loss.append(total_avg_loss)
	overall_ev.append(total_ev)
	overall_len.append(total_len)


ret = sum(overall_returns) / float(len(overall_returns))
p_win = sum(overall_prob_win) / float(len(overall_prob_win))
p_loss = sum(overall_prob_loss) / float(len(overall_prob_loss))
mean_win = sum(overall_avg_win) / float(len(overall_avg_win))
mean_loss = sum(overall_avg_loss) / float(len(overall_avg_loss))
avg_num_trades = sum(overall_len) / float(len(overall_len))
ev = sum(overall_ev) / float(len(overall_ev))

print 'Number of trades: {}'.format(avg_num_trades)
print 'Total returns over {} days:\t {:.2f}'.format( overall_days, ret )
print 'Total win probability over {} days:\t {:.2f}'.format( overall_days, p_win )
print 'Total loss probability over {} days:\t {:.2f}'.format( overall_days, p_loss )
print 'Total average win over {} days:\t {:.2f}'.format( overall_days, mean_win )
print 'Total average loss over {} days:\t {:.2f}'.format( total_days, mean_loss )
print 'Total expected value over {} days:\t {:.2f} \n'.format( overall_days, ev )
