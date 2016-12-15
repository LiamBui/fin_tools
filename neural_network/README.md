"# cs4701" 

To run, call "main.py" with arguments num_trials, activation_function_type, total_sample_set_size, num_test_points, num_hidden_layers, regularization_constant. Example: if you want to run 100 trials with the sigmoid function, a total sample set size of 10 points and 3 test points (10-3=7 training points), 3 hidden layers, and a regularization constant of 0.001, then your command line function is the following:

python main.py 100 sigmoid 10 3 3 0.001

To obtain data, call "download.py" with with arguments start_date, end_date. Must be formatted as specified. This will download data for the 500 constituents of the S&P 500, as well as the SPY ETF from the NYSE. Example: if you want data for the 500 constituents and SPY ETF between the dates 10/01/2016 and 10/08/2016, your command line function is the following:

python download.py 10012016 10082016
