#CS 4701

To run, call "main.py" with arguments num_trials, activation_function_type, total_sample_set_size, num_test_points, num_hidden_layers, regularization_constant. Example: if you want to run 100 trials with the sigmoid function, a total sample set size of 10 points and 3 test points (10-3=7 training points), 3 hidden layers, and a regularization constant of 0.001, then your command line function is the following:

python main.py 100 sigmoid 10 3 3 0.001

To obtain data, call "download.py" with with arguments start_date, end_date. Must be formatted as specified. This will download data for the 500 constituents of the S&P 500, as well as the SPY ETF from the NYSE. Example: if you want data for the 500 constituents and SPY ETF between the dates 10/01/2016 and 10/08/2016, your command line function is the following:

python download.py 10012016 10082016

#General Neural Network Information

You have input (xTr, yTr) where xTr is a num_data x num_features matrix, and yTr is a num_data x num_output_features matrix. xTr is the training input that you want to apply to your neural network for training, and yTr is the example output that it should provide. 

This feedforward neural network initializes random weights and random biases. It then applies the weights and biases for each hidden layer in the neural network through matrix multiplication (i.e., net = xTr \cdot w^T). It then applies an activation function, which introduces nonlinearity into the model. Typical activation functions include sigmoid and tanh, which are both S shaped curves and can both act as universal basis functions. In recent years, the rectifier function ReLU has gained more support, as it has been shown to outperform sigmoid and tanh in a variety of applications. We now sit at an output given weightings and biases. 

This output is compared to the true correct output (yTr) and is evaluated by a loss function. One simple example is \sum\limits_{i} |yhat_ - y_i| for each y_i in the training data. There are more advanced loss functions, such as hinge loss and root mean squared loss. In this application, we use 0.5 * \sum\limits_{i} (yhat_i - y)^2 or something. 

We then backpropagate a derivative of the loss with respect to weighting and biases, to find the direction in which the loss is moving. You want to minimize loss, so you want to take the opposite of the max derivative, which is simply the negative of the derivative of the loss with respect to weighting and with respect to the bias. Then, you update weights of the neural network model by the following: what = w + scalar * -dLossdWeight * w, or something. You find the minimum of the loss function by doing gradient descent, and in this example, we use the scipy optimize class, which has a method minimize. In this method, you can spit in a loss function and some inputs and outputs, and it will spit back arguments which provide minimal loss. Note that gradient descent may vary based on the max number of iterations and learning rate. Learning rate is done for you by the optimize class, but max iterations is a hyperparameter of the minimize function, so you should play around with the maximum number of iterations before convergence (i.e., a stopping criterion for convergence). 

After training is finished on each of the training examples, we now have a trained neural network model. This is basically a model that has weights and biases that minimize loss (i.e., difference in target output and predicted output). We are now ready to stick in any random test input we want. So we stick in xTe which is a num_examples x num_features matrix and our forward method from the neural network will provide an output based on optimal weightings and biases. The output gets denormalized and voila, you have a prediction. The prediction may either be regressive or a classification -- stock prices are inherently a regression problem, though we may also apply the neural network to be a classifier, which may, for example, predict direction of the move of a stock (i.e., tomorrow, will there be an uptick or downtick?). 

