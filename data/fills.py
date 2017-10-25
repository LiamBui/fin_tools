import pandas as pd
import datetime

LOCATION = './fills.csv'

def get_data():
	df = pd.read_csv(LOCATION)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	df = df.set_index(df[df.columns[0]])
	df.columns = ['trade_id','product','side','datetime','size','size_unit','price','fee','total','unit']

	return df

def update_avg_entry(avg_entry, available_btc, buy_price, buy_size):
	return (avg_entry * available_btc + (buy_size * buy_price)) / (available_btc + float(buy_size))

def main():
	df = get_data()

	wins = []
	losses = []
	last_side_buy = False
	avg_entry = 0
	available_btc = 0

	for idx, row in df.iloc[1:].iterrows():
		if last_side_buy:
			# buy consecutive time 
			if row['side'] == 'BUY':
				avg_entry = update_avg_entry(avg_entry, available_btc, row['price'], row['size'])
				available_btc += row['size']
			# first sell after buying
			elif row['side'] == 'SELL':
				last_side_buy = False
				if row['price'] > avg_entry:
					wins.append(row['price'] - avg_entry)
					available_btc = 0 
				else:
					losses.append(row['price'] - avg_entry)
					available_btc = 0
		else:
			# first buy after selling
			if row['side'] == 'BUY':
				last_side_buy = True
				avg_entry = update_avg_entry(avg_entry, available_btc, row['price'], row['size'])
				available_btc += row['size']
			# sell consecutive time, shouldn't happen
			elif row['side'] == 'SELL':
				if row['price'] > avg_entry:
					available_btc = 0
				else:
					available_btc = 0

	print 'Num wins:\t {}'.format(len(wins))
	print 'Num losses:\t {}'.format(len(losses))

	prob_win = len(wins) / float(len(wins) + len(losses))
	prob_lose = len(losses) / float(len(wins) + len(losses))
	print 'Percent wins:\t {}'.format(prob_win)
	print 'Percent losses:\t {}'.format(prob_lose)

	avg_win = sum(wins) / float(len(wins))
	avg_loss = sum(losses) / float(len(losses))

	print 'Average win:\t {}'.format(avg_win)
	print 'Average loss:\t {}'.format(avg_loss)
	print 'Expected value:\t {}'.format(avg_win * prob_win + avg_loss * prob_lose)

if __name__ == '__main__':
	main()