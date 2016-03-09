import numpy as np
import theano
import theano.tensor as T

class LinearRegression(object):
    """
    The Linear Regression layer for the final output of the MLP. It's similar to LogisticRegression,
    but we will only have one output layer, and we don't use their 'errors' method.
    """

    def __init__(self, input, n_in, n_out):
        """
        :input: A symbolic variable that describes the input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value = np.zeros( (n_in, n_out), dtype=theano.config.floatX ),
                               name = 'W',
                               borrow = True)
        self.b = theano.shared(value = np.zeros( (n_out,), dtype=theano.config.floatX ),
                               name = 'b',
                               borrow = True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.p_y_given_x[:,0]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

    def squared_errors(self, y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        return T.mean((self.y_pred - y) ** 2)

class HiddenLayer(object):
    """
    Hidden Layer class for a Multi-Layer Perceptron. This is exactly the same as the reference
    code from the documentation, except for T.sigmoid instead of T.tanh.
    """

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        :rng: A random number generator for initializing weights.
        :input: A symbolic tensor of shape (n_examples, n_in).
        :n_in: Dimensionality of input.
        :n_out: Number of hidden units.
        :activation: Non-linearity to be applied in the hidden layer.
        """

        # W is initialized with W_values, according to the "Xavier method".
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize the bias weights.
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # The output of all the inputs, "squashed" via the activation function.
        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

def load_data(dataset):
    data = []
    import re

    if isinstance(dataset, str) and re.search('\.csv$', dataset):
        with open(dataset) as f:
            f.readline()
            for line in f:
                if line=='\n':
                    continue
                data.append([float(x.strip()) for x in line.strip().split(',')])

        train_x = [data[x][:-1] for x in range(int(len(dataset)*.7))]
        train_y = [data[x][-1] for x in range(int(len(dataset)*.7))]

        valid_x = [data[x][:-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]
        valid_y = [data[x][-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]

        test_x = [data[x][:-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]
        test_y = [data[x][-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]

        train_set = (np.array(train_x), np.array(train_y))
        valid_set = (np.array(valid_x), np.array(valid_y))
        test_set = (np.array(test_x), np.array(test_y))

    else:
        """
        Copying this from documentation online, including some of the nested 'shared_dataset' function,
        but I'm also returning the number of features, since it's easiest to detect that here.
        """
        train_set, valid_set,test_set = dataset[0], dataset[1], dataset[2]
        assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], \
            "Number of features for train,val do not match: {} and {}.".format(train_set.shape[1],valid_set.shape[1])


    def shared_dataset(data_xy, borrow=True):
            """
            Function that loads the dataset into shared variables. It is DIFFERENT from the online
            documentation since we can keep shared_y as floats; we won't be needing them as indices.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    num_features = (train_set[0].shape)[1]
    rval = [(train_set_x,train_set_y), (valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return rval,num_features

