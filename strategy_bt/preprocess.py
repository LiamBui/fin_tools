'''
Files obtained from https://api.bitcoincharts.com/v1/csv/
'''

import pandas as pd
import datetime, random

def prompt():
	exchange_name = raw_input('Input exchange name: ')
	date = raw_input('Input date of file: ')
	return exchange_name, date

def get_data(exchange_name, date):
	#exchange_name, date = prompt()
	folder_name = exchange_name + 'USD.csv.' + date + '/'
	file_name = '.' + exchange_name + 'USD.csv'
	location = '../data/' + folder_name + file_name

	df = pd.read_csv(location)
	df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], unit='s')

	df = df.set_index(df[df.columns[0]])
	df = df.drop(df.columns[[0]], axis=1)
	df.columns = ['price','size']
	df.index.name = 'datetime'

	return df

def generate_train_test_split(df, sample_size, train_size=0.8, test_size=0.2):
	# exchange_name, date = prompt()

	# # default to taking train_size 
	# if train_size + test_size != 1: 
	# 	train_size, test_size = train_size, 1-train_size

	# df = get_data(exchange_name, date)

	seed = random.randint(0, len(df.index)-sample_size-1)
	sample = df.iloc[seed : seed+sample_size]
	while (sample.iloc[0].name < datetime.datetime(2016,1,1)):
		seed = random.randint(0, len(df.index)-sample_size-1)
		sample = df.iloc[seed : seed+sample_size]

	num_train = train_size * len(sample.index)
	num_test = len(sample.index) - num_train

	print num_train, num_test

	train, test = sample.tail(n=int(num_train)), sample.head(n=int(num_test))

	return train, test

