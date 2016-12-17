import pandas as pd
import random, math

'''
Range: (-infty, infty)
'''
def normalization_gaussian(df):
	return (df - df.mean()) / (df.std())
def denormalization_gaussian(df, df_prime):
	return df_prime * df.std().iloc[0] + df.mean().iloc[0]

'''
Range: minmax
'''
def normalization_minmax(df, minmax=(0,1)):
	minimum, maximum = minmax
	return ( (df - df.min()) / (df.max() - df.min()) ) * (maximum - minimum) + minimum
def denormalization_minmax(df, df_prime, minmax=(0,1)):
	minimum, maximum = minmax
	return (df_prime - minimum) / (maximum - minimum) * (df.max().iloc[0] - df.min().iloc[0]) + df.min().iloc[0]

def generate_train_test(location, symb, sample_size):
	if location[len(location)-1] == '/':
		location = location + symb + '.csv'
	else:
		location = location + '/' + symb + '.csv'
	df = pd.read_csv(location, delimiter='\t')
	df = df.iloc[::-1]
	df = df.set_index(len(df.index) - df.index)

	train_sets = []
	test_sets = []

	for idx, row in df.iloc[ : len(df.index) - 2].iterrows():
		if df.iloc[idx + 1]['Open'] >= row['Close']:
			df = df.set_value(idx, 'Label', 1)
		else:
			df = df.set_value(idx, 'Label', -1)

	for idx in range(len(df.index)-sample_size-1):
		train_set = df.iloc[idx : idx + sample_size]
		test_set = df.iloc[idx + sample_size]
		train_sets.append(train_set)
		test_sets.append(test_set)

	return zip(train_sets, test_sets)

def generate_xs_ys(train, test):
	xTr = train[['Open','High','Low','Close','Volume']]
	yTr = train[['Label']]
	xTe = test[['Open','High','Low','Close','Volume']]
	yTe = test[['Label']]

	return xTr, yTr, xTe, yTe

def evaluate(output, yTe):
	test_results = zip(output, yTe)
	return sum(int(x == y) for (x,y) in test_results) / float(len(yTe))
