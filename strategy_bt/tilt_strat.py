import pandas as pd
import matplotlib.pyplot as plt

def get_data():
	location = '../data/output.log'
	df = pd.read_csv(location)

	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
	#df = df.set_index(df[df.columns[0]])
	#df = df.drop(df.columns[[0]], axis=1)
	df.columns = ['datetime','tilt','tilt_up','price_delta','bid','ask','midprice']
	#df.index.name = 'datetime'

	return df

def graph(df):
	indicators = get_indicators(df)
	xs = [x[0] for x in indicators]
	ys = [x[1] for x in indicators]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(df['datetime'], df['midprice'], color='red', label='MidPrice')
	ax1.scatter(xs, ys, color='green', label='Indicators')

	plt.legend(loc='upper left')
	plt.ylabel('Y values')
	plt.xlabel('X values')
	plt.show()

def run(df):
	(idxs, midprices) = zip(*get_indicators(df))
	available_btc = 0.0 
	pnl = 0.0
	funds = 1000.0
	for idx, row in df.iterrows():
		if idx in idxs:
			available_btc += 0.01
			funds -= row['ask']
			pnl -= row['ask']
		elif row['tilt'] <= 0.1:
			available_btc -= 0.01
			funds += row['bid']
			pnl += row['bid']
		else:
			pass
	print 'PnL : ${:.2f}\t Funds : ${:.2f}\t Available BTC : {:.8f}'.format(pnl, funds, available_btc)
	return

def get_indicators(df):
	indicators = []

	for idx, row in df.iloc[25:].iterrows():
		if df.iloc[idx - 25 : idx]['tilt_up'].sum() >= 25 and row['tilt'] >= 0.9:
			indicators.append((df.iloc[idx]['datetime'], df.iloc[idx]['midprice']))
	return indicators

def run_strat(df):
	pass

def main():
	df = get_data()
	run(df)
	graph(df)

if __name__ == '__main__':
	main()