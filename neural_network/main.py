import numpy as np
import neural_network, trainer, preprocess, sys
import matplotlib.pyplot as plt

def main(argv):
	evaluations = []
	accuracies = []
	iterations = int(argv[0])
	func_type = argv[1]
	sample_size = int(argv[2])
	test_size = int(argv[3])
	hidden_layers = int(argv[4])
	regularization = float(argv[5])
	for i in range(iterations):
		train, test = preprocess.generate_train_test('../data/data', 'SP500',sample_size, test_size)
		train = train.iloc[np.random.permutation(len(train))]
		test = test.iloc[np.random.permutation(len(test))]
		xTr, yTr, xTe, yTe, raw_yTr, raw_yTe = preprocess.generate_xs_ys(train, test)

		# Normalize data based on activation function used.
		_, normalized_xTr = preprocess.normalize(xTr, func_type)
		original_yTr, normalized_yTr = preprocess.normalize(yTr, func_type)
		_, normalized_xTe = preprocess.normalize(xTe, func_type)
		original_yTe, normalized_yTe = preprocess.normalize(yTe, func_type)

		# Adjust types
		normalized_xTr = normalized_xTr.values
		normalized_yTr = normalized_yTr.values
		normalized_xTe = normalized_xTe.values
		normalized_yTe = normalized_yTe.values

		# Create and train the neural network.
		neural_net = neural_network.Neural_Network(len(normalized_xTr), [5,hidden_layers,1], func_type, regularization=regularization)
		T = trainer.Trainer(neural_net)
		T.train(normalized_xTr, normalized_yTr, normalized_xTe, normalized_yTe)

		# Get predictions.
		output = neural_net.forward(normalized_xTe)

		# Denormalize predictions.
		denormalized_output = preprocess.denormalize(original_yTr, output, func_type)

		# Evaluate.
		evaluation = preprocess.evaluate(denormalized_output, yTe.values)
		evaluations.append(evaluation)
		accuracy = preprocess.accuracy(denormalized_output, yTe.values)
		accuracies.append(accuracy)

		print 'Iteration {0} / {2} \t Accuracy: {1} \t\t Accurate within 1%: {3}\n'.format(i+1, accuracy, iterations, evaluation)

	print 'Average binary evaluation: ', sum(evaluations) / float(len(evaluations))
	print 'Average accuracy: ', sum(accuracies) / float(len(accuracies))

if __name__ == '__main__':
	main(sys.argv[1:])