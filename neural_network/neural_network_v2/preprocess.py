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

def generate_train_test(location, symb, chunk_size, test_size):
	if location[len(location)-1] == '/':
		location = location + symb + '.csv'
	else:
		location = location + '/' + symb + '.csv'
	df = pd.read_csv(location, delimiter='\t')

	df = df.set_index('Date')

	seed = random.randint(0, len(df.index)-(chunk_size+1))
	sample = df.iloc[seed : seed+chunk_size]
	a = len(sample.index) - test_size
	b = len(sample.index) - a
	train,test = sample.tail(n=a), sample.head(n=b)

	return train,test

def generate_xs_ys(train, test):
	raw_yTr = train[['Label']]
	raw_yTe = test[['Label']]

	xTr = train[['Open','High','Low','Close','Volume']]
	yTr = train[['Label']]
	xTe = test[['Open','High','Low','Close','Volume']]
	yTe = test[['Label']]

	return xTr, yTr, xTe, yTe, raw_yTr, raw_yTe

def normalize(df,func_type):
	if func_type.lower() == 'sigmoid':
		return df, normalization_minmax(df, minmax=(0,1))
	elif func_type.lower() == 'tanh':
		return df, normalization_minmax(df, minmax=(-1,1))
	elif func_type.lower() == 'relu':
		return df, normalization_gaussian(df)
	else:
		return df, normalization_gaussian(df)

def denormalize(df, df_prime, func_type):
	if func_type.lower() == 'sigmoid':
		return denormalization_minmax(df, df_prime, minmax=(0,1))
	elif func_type.lower() == 'tanh':
		return denormalization_minmax(df, df_prime, minmax=(-1,1))
	elif func_type.lower() == 'relu':
		return denormalization_gaussian(df, df_prime)
	else:
		return denormalization_gaussian(df, df_prime)

def evaluate(output, yTe):
	test_results = zip(output, yTe)
	return sum(int(math.fabs(x-y) < 0.01 * y) for (x,y) in test_results) / float(len(yTe))

def accuracy(output, yTe):
	test_results = zip(output, yTe)
	return sum([1-(math.fabs(y-x)/float(y)) for (x,y) in test_results]) / len(test_results)