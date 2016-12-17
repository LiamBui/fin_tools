#Price Directional Predictions using SVMs

To run, call "main.py" with arguments kernel_type and train_size, simply call:

python main.py kernel_type train_size

Valid kernel types include linear, poly, rbf, or sigmoid. Note that in order to train an SVM, you must include in the training data at least two label classifications. In other words, if the entirety of the training data is all of the same label type, you will not be able to train. 

#General SVM information

The general intuition is that the SVM creates a loss-minimizing hyperplane between two classes using training data. It then tests by determining which side of the hyperplane your test data lies. Typically, SVM's are useful when there are fewer data points than there are features and become less useful as the number of features becomes far greater than the number of data points. 

#TODO:

Figure out normalization of data

Figure out svm.SVC.score() method
