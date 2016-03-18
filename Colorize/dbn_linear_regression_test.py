from DBN_linear import test_DBN, predict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from my_utility import tic, toc
from sklearn.utils import shuffle
from main_colorize import create_data


def load_saved_data(filename):
    tic()
    f = h5py.File(filename, 'r')
    X = f['/data/X'][:]
    Y = f['/data/Y'][:]
    norm_y = f['/data/norm_y'][:]
    # norm_c= f['/data/norm_c'][:]

    f.close()
    toc("Data loaded from: " + filename)
    return X, Y, norm_y


def train(load_from_file):
    if load_from_file == 1:
        X, Y, norm = load_saved_data('data_YCrCb_normalized.h5')
    else:
        X, Y, norm = create_data(folder='/home/exx/PycharmProjects/Train_Imgs/', num_samples=30, save=0,output_filename = 'data_YCrCb_normalized.h5')

    num_train_images = 26
    num_train = num_train_images * 224 * 224
    num_val = num_train + int(3) * 224 * 224#int((20 - num_train_images) / 2) * 224 * 224

    X_train = X[0:num_train, :]
    Y_train = Y[0:num_train, :]

    X_train, Y_train = shuffle(X_train, Y_train, random_state=1)

    X_val = X[num_train:num_val, :]
    Y_val = Y[num_train:num_val, :]

    X_test = X[num_val:, :]
    Y_test = Y[num_val:, :]

    data = [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]

    test_DBN(finetune_lr=0.1, pretraining_epochs=3, L1_reg=0.00, k=1,
             pretrain_lr=0.01, training_epochs=1000, L2_reg=0.0001,

             dataset=data, batch_size=2048, layer_sizes=[100, 100, 100], output_classes=2)

    return


def predictor(load_from_file):
    _, _, norm = load_saved_data('data_YCrCb_normalized.h5')

    num_samples = 10
    if load_from_file == 1:
        X, Y, _ = load_saved_data('separate_test_YCrCb.h5')
    else:
        X, Y, _ = create_data(num_samples, folder='/home/exx/PycharmProjects/Test_Imgs/', save=1,output_filename = 'separate_test_YCrCb.h5')

    max_x = norm[1]
    min_x = norm[0]
    d = (max_x - min_x)
    indices = [i for i, x in enumerate(d) if x == 0]
    d[indices] = 1
    Z = (X - min_x) / d
    Z[:, 0] = X[:, 0]

    output = predict(Z, filename='best_model_actual_data.pkl')

    Y_channel = X[:, 0:1]
    Y_channel = Y_channel.reshape(num_samples, 224, 224, 1)
    CrCb_out = output.reshape(num_samples, 224, 224, 2)
    out_imgs = np.concatenate((Y_channel, CrCb_out), axis=3).astype(np.float32)

    for i in range(num_samples):
        output_img = (cv2.cvtColor(out_imgs[i, :, :, :], cv2.COLOR_YCR_CB2BGR)) * 255
        output_folder = "/home/exx/PycharmProjects/Output_Imgs/"
        cv2.imwrite(output_folder + "out_" + str(i) + ".jpg", output_img)

    # cv2.imshow('disp',output_img)

    print("Prediction Completed. Images saved at: " + output_folder)

    # test_DBN(finetune_lr=0.1, pretraining_epochs=5,
    #          pretrain_lr=0.01, k=1, training_epochs=50,
    #          #dataset='mnist.pkl.gz', batch_size=10,
    #          dataset='randomdata.csv', batch_size=1,
    #          #n_ins=784, layer_sizes=[100, 50,20], output_classes=10,
    #          n_ins=4, layer_sizes=[10, 10,10], output_classes=10,
    #          load_from=None, save_to=None)
    return


if 0:
    train(load_from_file=0)
else:
    predictor(load_from_file=0)
