from DBN_linear import test_DBN, predict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from my_utility import tic, toc
from sklearn.utils import shuffle


def load_data(filename):
    tic()
    f = h5py.File(filename, 'r')
    X = f['/data/X'][:]
    Y = f['/data/Y'][:]
    norm_y= f['/data/norm_y'][:]
    # norm_c= f['/data/norm_c'][:]

    f.close()
    toc("Data loaded from file.")
    return X,Y,norm_y

def train():
    X,Y,norm= load_data('data_YCrCb_normalized.h5')

    #X, Y = shuffle(X, Y, random_state=1)
    num_train_images = 16
    num_train=num_train_images*224*224
    num_val = num_train+int((20-num_train_images)/2)*224*224

    X_train = X[0:num_train, :]
    Y_train = Y[0:num_train, :]

    X_val = X[num_train:num_val, :]
    Y_val = Y[num_train:num_val, :]

    X_test = X[num_val:, :]
    Y_test = Y[num_val:, :]


    data = [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]

    test_DBN(finetune_lr=0.001, pretraining_epochs=3, L1_reg=0.00, k=1,
             pretrain_lr=0.01, training_epochs=1000, L2_reg=0.0001,

             dataset=data, batch_size=1024, layer_sizes=[100,100], output_classes=2)


    return

def predictor():
    _,_,norm= load_data('data_YCrCb_normalized.h5')
    X, Y,_=load_data('separate_test_YCrCb.h5')

    max_x=norm[1]
    min_x=norm[0]
    d=(max_x-min_x)
    indices = [i for i, x in enumerate(d) if x == 0]
    d[indices]=1
    Z =(X-min_x)/d
    Z[:,0]=X[:,0]

    output = predict(Z, filename='best_model_actual_data.pkl')

    Y_channel= X[:,0:1]
    Y_channel=Y_channel.reshape(10,224,224,1)
    CrCb_out=output.reshape(10,224,224,2)
    out_imgs=np.concatenate((Y_channel,CrCb_out),axis=3).astype(np.float32)

    output_img = (cv2.cvtColor(out_imgs[0,:,:,:], cv2.COLOR_YCR_CB2BGR))
    cv2.imshow('disp',output_img)

    print('Completed.')

    # test_DBN(finetune_lr=0.1, pretraining_epochs=5,
    #          pretrain_lr=0.01, k=1, training_epochs=50,
    #          #dataset='mnist.pkl.gz', batch_size=10,
    #          dataset='randomdata.csv', batch_size=1,
    #          #n_ins=784, layer_sizes=[100, 50,20], output_classes=10,
    #          n_ins=4, layer_sizes=[10, 10,10], output_classes=10,
    #          load_from=None, save_to=None)
    return

predictor()