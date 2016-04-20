'''
For detailed documentation, check the deeplearning.net tutorial code.
'''
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gof import graph
import matplotlib.pyplot as plt
from rbm import RBM


class SemiLastLayer(object):
    def __init__(self, input, n_in, n_out):
        """
        :input:  input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.p_y_given_x[:, :]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

    def squared_errors(self,y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        return  (T.mean(T.sqr(self.y_pred - y),axis=0))

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class HiddenLayer(object):
    """
    Hidden Layer class for a Multi-Layer Perceptron. This is exactly the same as the reference
    code from the documentation, except for T.sigmoid instead of T.tanh.
    """

    def __init__(self, rng, input, n_in, n_out,dropout_rate, W=None, b=None, activation=T.tanh):
        """
        :rng: A random number generator for initializing weights.
        :input: A symbolic tensor of shape (n_examples, n_in).
        :n_in: Dimensionality of input.
        :n_out: Number of hidden units.
        :activation: Non-linearity to be applied in the hidden layer.
        """

        # W is initialized with W_values, according to the "Xavier method".
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize the bias weights.
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # The output of all the inputs, "squashed" via the activation function.
        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.output[:, :]


    def squared_errors(self, y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        return  (T.mean(T.sqr(self.y_pred - y),axis=0))
        # return T.mean(T.sqr(self.y_pred - y))

def convert_dataset(dataset):
    train_set, valid_set, test_set = dataset[0], dataset[1], dataset[2]
    assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], \
        "Number of features for train,val do not match: {} and {}.".format(train_set.shape[1], valid_set.shape[1])

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    num_features = (train_set[0].shape)[1]
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval, num_features

class DBN(object):
    def __init__(self, numpy_rng, n_ins,
                 hidden_layers_sizes, n_outs, L1_reg, L2_reg, dropout_rate, theano_rng=None):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as a matrix
        self.y = T.matrix('y')  # the labels are presented as 1D vector
        # of labels

        for i in range(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        dropout_rate=dropout_rate,
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # self.decisionLayer = HiddenLayer(rng=numpy_rng,input=self.sigmoid_layers[-1].output,
        #                              n_in=hidden_layers_sizes[-1],
        #                              n_out=n_outs,
        #                              activation=T.nnet.relu)

        self.decisionLayer = SemiLastLayer(input=self.sigmoid_layers[-1].output,
                                     n_in=hidden_layers_sizes[-1],
                                     n_out=n_outs,
                                     )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.decisionLayer.W).sum()
        for i in range(2 * self.n_layers)[0::2]:
            self.L1 += abs(self.params[i]).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.decisionLayer.W ** 2).sum()
        for i in range(2 * self.n_layers)[0::2]:
            self.L2_sqr += (self.params[i] ** 2).sum()

        self.squared_errors = self.decisionLayer.squared_errors

        self.finetune_cost = (self.squared_errors(self.y)).sum()+ L1_reg * self.L1 + L2_reg * self.L2_sqr

        self.params.extend(self.decisionLayer.params)
        self.input = input

        self.y_pred = self.decisionLayer.y_pred

    def pretrain_setup(self, train_set_x, batch_size, k):

        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def finetune_setup(self, datasets, batch_size):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar()  # index to a [mini]batch
        lr = T.scalar('lr', dtype=theano.config.floatX)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        train_fn = theano.function(
            inputs=[index,lr],
            outputs=[self.finetune_cost, self.squared_errors(self.y)],
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }

        )

        test_score_i = theano.function(
            [index],
            self.squared_errors(self.y),
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.squared_errors(self.y),
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(int(n_valid_batches))]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(int(n_test_batches))]

        return train_fn, valid_score, test_score

def test_DBN(finetune_var, pretrain_var, L1_reg,L2_reg, dataset, batch_size, layer_sizes, output_classes):
    datasets, features = convert_dataset(dataset)

    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    if finetune_var.retrain==False:
        numpy_rng = numpy.random.RandomState(125)
        print('... building the model')

        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, n_ins=features,
                  hidden_layers_sizes=layer_sizes,
                  n_outs=output_classes,dropout_rate=0.3,
                  L1_reg=L1_reg, L2_reg=L2_reg)

        # setup the pretraining
        print('... getting the pretraining functions')
        pretraining_fns = dbn.pretrain_setup(train_set_x=train_set_x, batch_size=batch_size, k=pretrain_var.k)
        dbn= pretrain(pretraining_fns, pretrain_var, n_train_batches, dbn)
    else:
        model_file = open(finetune_var.model_in, 'rb')
        dbn = pickle.load(model_file)
        model_file.close()

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, valid_score, test_score = dbn.finetune_setup(
            batch_size=batch_size,
            datasets=datasets)

    dbn=train(n_train_batches,finetune_var,train_fn, valid_score, test_score,dbn)

    # save the last model
    with open('final' + finetune_var.model_name, 'wb') as f2:
        pickle.dump(dbn, f2, protocol=pickle.HIGHEST_PROTOCOL)
        print('Last Model Saved as: ' + 'final' + finetune_var.model_name)
    return

def pretrain(pretraining_fns, pretrain_var, n_train_batches, dbn):

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretrain_var.epochs):
            # go through the training set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_var.lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), )
            print(numpy.mean(c))

    end_time = timeit.default_timer()

    print(('The pretraining code in file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return dbn

def train(n_train_batches,finetune_var,train_fn, valid_score, test_score,dbn):
    print('... finetuning the model')
    patience = 20000 #4 * n_train_batches
    patience_increase = 5000
    improvement_threshold = 0.995
    validation_frequency = int(min(n_train_batches, patience / 2))

    best_validation_loss = numpy.inf
    test_error = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < finetune_var.epochs) and (not done_looping):
        epoch = epoch + 1
        # finetune_lr=finetune_lr
        for minibatch_index in range(int(n_train_batches)):

            minibatch_finetune_cost, minibatch_mse_cost = train_fn(minibatch_index,finetune_var.lr,)
            iter = (epoch - 1) * int(n_train_batches) + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = valid_score()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation MSE: %f , finetune_cost: %f, train_mse: %f'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss,
                        minibatch_finetune_cost, numpy.mean(minibatch_mse_cost)
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_score()
                    test_error = numpy.mean(test_losses)
                    print(('     \tepoch %i, minibatch %i/%i, test MSE: '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_error))

                    # save the best model
                    with open(finetune_var.model_name, 'wb') as f:
                        pickle.dump(dbn, f, protocol=pickle.HIGHEST_PROTOCOL)


            if patience <= iter:
                done_looping = True

                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f, '
            'with test performance %f'
        ) % (best_validation_loss, test_error)
    )
    print(('The fine tuning code in file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time)
                               / 60.)), file=sys.stderr)
    print('Best Model Saved as: ' + finetune_var.model_name)
    return dbn

def predict(X_test, filename='best_model_actual_data.pkl'):
    # load the saved model
    model_file = open(filename, 'rb')
    classifier = pickle.load(model_file)
    model_file.close()
    y_pred = classifier.y_pred

    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']
    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)

    predicted_values = predict_model(X_test.astype(numpy.float32))

    return predicted_values

if __name__ == '__main__':
    X_train = numpy.random.rand(2500, 2) * .4
    X_train = numpy.append(X_train,numpy.random.rand(2500, 2) * .4 + .6,axis=0)
    X_train = numpy.append(X_train, numpy.random.rand(5000, 2) * .4+.3, axis=0)

    y_train = numpy.random.rand(5000, 2) * .3 + .7
    y_train = numpy.append(y_train, numpy.random.rand(5000, 2) * .3 , axis=0)

    X_val = numpy.random.rand(500, 2) * .4
    X_val = numpy.append(X_val,numpy.random.rand(500, 2) * .4 + .6,axis=0)
    X_val = numpy.append(X_val, numpy.random.rand(1000, 2) * .4+.3, axis=0)

    y_val = numpy.random.rand(1000, 2) * .3+ .7
    y_val = numpy.append(y_val, numpy.random.rand(1000, 2) * .3 , axis=0)

    X_test = numpy.random.rand(500, 2) * .4
    X_test = numpy.append(X_test,numpy.random.rand(500, 2) * .4 + .6,axis=0)
    X_test = numpy.append(X_test, numpy.random.rand(1000, 2) * .4+.3, axis=0)

    y_test = numpy.random.rand(1000, 2) * .3+ .7
    y_test = numpy.append(y_test, numpy.random.rand(1000, 2) * .3 , axis=0)

    data = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    print('Need to figureout if self.y = T.matrix or T.vector is correct for the problem')

    class pretrain_var(object):
        def __init__(self, lr, epochs, k):
            self.lr = lr
            self.epochs = epochs
            self.k = k
            self.n_train_batches = None
            self.pretraining_fns = None
    class finetune_var(object):
        def __init__(self, lr, epochs, model_out_filename):
            self.lr = lr
            self.epochs = epochs
            self.model_out_filename = model_out_filename


    pretrain_var.lr = 0.001
    pretrain_var.epochs = 50
    pretrain_var.k = 1

    finetune_var.lr = 0.1
    finetune_var.epochs = 5
    finetune_var.model_out_filename = 'test_model'

    test_DBN(finetune_var,
             pretrain_var,
             L1_reg=0.000,
             L2_reg=0.000001,
             dataset=data,
             batch_size=1000,
             layer_sizes=[10, 10, 10],
             output_classes=2)

    out=predict( X_test,filename='test_model')

    plt.scatter(X_test[:,0],out[:,0])
    plt.show()
    plt.scatter(X_test[:,0],y_test[:,0])
    # hist, bins = numpy.histogram(out, bins=100)
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.show()
    # print((y_test-out) ** 2).mean(axis=None)
