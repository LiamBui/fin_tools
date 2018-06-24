import numpy as np
import pandas as pd
import sys, preprocess, tools, liquidity_strat

# Wrapper function for getting data and computing statistics
def main():
	df1 = preprocess.get_data('coinbase','20170203')
	df2 = preprocess.get_data('bitfinex','20170203')

	trades = liquidity_strat.run(df1, df2)
	tools.compute_statistics(trades)

	return trades

if __name__ == '__main__':
	main()
