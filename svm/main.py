import numpy as np
import svm, preprocess, sys

def main(argv):
	evaluations = []
	scores = []
	kernel_type = argv[0]
	test_size = int(argv[1])

	for train, test in preprocess.generate_train_test('../data/data', 'SP500', test_size):
		try:
			xTr, yTr, xTe, yTe = preprocess.generate_xs_ys(train, test)

			# normalized_xTr = preprocess.normalization_gaussian(xTr).values
			# normalized_yTr = yTr.values
			# normalized_xTe = preprocess.normalization_gaussian(xTe).values
			# normalized_yTe = yTe.values

			normalized_xTr = xTr.values
			normalized_yTr = yTr.as_matrix().ravel()
			normalized_xTe = xTe.reshape(1,5)
			normalized_yTe = yTe.as_matrix().ravel()

			model = svm.SVM(kernel_type, normalized_xTr, normalized_yTr)
			model.train()
			predictions = model.predict(normalized_xTe)
			# score = model.score(normalized_xTe, normalized_yTe)

			evaluations.append(preprocess.evaluate(predictions, normalized_yTe))
			# scores.append(score)
		except:
			continue

	print 'Mean correct: {0}'.format( sum(evaluations)/float(len(evaluations)) )

if __name__ == '__main__':
	main(sys.argv[1:])