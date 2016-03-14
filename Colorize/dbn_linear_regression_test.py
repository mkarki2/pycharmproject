from DBN_linear import test_DBN, predict
import numpy as np
import matplotlib.pyplot as plt
import h5py
from my_utility import tic, toc
from sklearn.utils import shuffle

train = 1
prediction = 0

tic()
f = h5py.File('data_YCrCb_normalized.h5', 'r')
X = f['/data/X'][:]
Y = f['/data/Y'][:]
norm_y= f['/data/norm_y'][:]
norm_c= f['/data/norm_c'][:]

f.close()
toc("Data loaded from file.")

train_fraction=0.8
#X, Y = shuffle(X, Y, random_state=1)
num_train = int(train_fraction * len(X))
num_val = int((1-train_fraction)/2 * len(X)) + num_train

X_train = X[0:num_train, 0:1]
Y_train = Y[0:num_train, :]

X_val = X[num_train:num_val, 0:1]
Y_val = Y[num_train:num_val, :]

X_test = X[num_val:, 0:1]
Y_test = Y[num_val:, :]

if train == 1:
    data = [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]

    test_DBN(finetune_lr=0.001, pretraining_epochs=1, L1_reg=0.00, k=1,
             pretrain_lr=0.01, training_epochs=1000, L2_reg=0.0001,

             dataset=data, batch_size=4096, layer_sizes=[100,100], output_classes=2)
if prediction == 1:
    output = predict(X_test, filename='best_model_actual_data.pkl')
    print("Predicted values for the some examples in test set:")
    plt.hist(output)
    plt.show()
    print(output[990:1010])

    # test_DBN(finetune_lr=0.1, pretraining_epochs=5,
    #          pretrain_lr=0.01, k=1, training_epochs=50,
    #          #dataset='mnist.pkl.gz', batch_size=10,
    #          dataset='randomdata.csv', batch_size=1,
    #          #n_ins=784, layer_sizes=[100, 50,20], output_classes=10,
    #          n_ins=4, layer_sizes=[10, 10,10], output_classes=10,
    #          load_from=None, save_to=None)
