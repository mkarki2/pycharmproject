import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from my_utility import tic, toc
from main_colorize import create_data
# from DBN_fa import test_DBN, predict
from DBN_dropout import test_DBN, predict
import theano.sandbox.cuda

theano.sandbox.cuda.use("gpu0") #1,0 in use
###
###
###

test_flag=0  #0: TRAIN 1: TEST

###
###
###
class pretrain_var(object):
    def __init__(self, lr, epochs,k):
        self.lr= lr
        self.epochs = epochs
        self.k= k

class finetune_var(object):
    def __init__(self, lr, epochs,model_out_filename):
        self.lr= lr
        self.epochs = epochs
        self.model_out_filename= model_out_filename
        self.model_in = None
        self.retrain = False

num_samples =   24    #12
num_train   =   18    #8
num_val     =   3     #2
save        =   1     #1= yes, 0 - no
train_data_file ='train_data5.h5'
train_folder    ='/home/exx/PycharmProjects/Train_Imgs/'
model_save      ='model50.pkl'

#44:layer_sizes=[160,80,40,20], pr = 0.001 pe=3 fl=0.07
#45:layer_sizes=[160,80,40,20], pr = 0.001 pe=3 fl=0.05
#46:same + L2_reg = 0.000001 fl=0.07
#49:L2_reg=0.00001,             layer_sizes=[1600,320,80,40,20], decisionlayer = not hidden layer
#x decision layer = hidden layer, 0.05
#x pretrain_lr=0.0001
#49_1:L2_reg=0.00001,             layer_sizes=[1600,320,80,40,20], decisionlayer = not hidden layer model49 retrained F_lr     = 0.1
#50 Dropout, Finetune epochs = 10000

pretrain_var.lr     = 0.0001
pretrain_var.epochs = 20
pretrain_var.k      = 1

finetune_var.lr     = 0.1
finetune_var.epochs = 10000
finetune_var.model_name = model_save
finetune_var.retrain=False
finetune_var.model_in='finalmodel50.pkl'

num_test_samples = 12
test_output_filename    ='test_data5.h5'
test_folder             ='/home/exx/PycharmProjects/Test_Imgs/'

reconstruct_image = 1 #1= yes, 0 - no
output_folder           = '/home/exx/PycharmProjects/Output_Imgs/'

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
        X, Y, norm = load_saved_data(train_data_file)
    else:
        X, Y, norm = create_data(num_samples, train_folder, save,
                                 train_data_file, test_flag)

    data = divide_data(X, Y, num_train, num_val)

    # Good Results with this set of hyperparameters

    # test_DBN(finetune_lr=0.07, pretraining_epochs=2, L1_reg=0.000, k=1,
    #          pretrain_lr=0.001, training_epochs=100000, L2_reg=0.00001,
    #          model_out_filename=model_save,
    #          dataset=data, batch_size=100352, layer_sizes=[80,40,20], output_classes=2)
    test_DBN(finetune_var,
             pretrain_var,
             L1_reg=0.000,
             L2_reg=0.00001,
             dataset=data,
             batch_size=50176*num_val,
             layer_sizes=[1600,320,80,40,20],
             output_classes=2)

    return

def predictor(load_from_file):

    X_all, Y_all, norm = load_saved_data(train_data_file)

    data = divide_data(X_all, Y_all, num_train, num_val)
    X_, Y_ = data[0]


    if load_from_file == 1:
        X, Y, _ = load_saved_data(test_output_filename)
    else:
        X, Y, _ = create_data(num_test_samples, test_folder, save,
                              test_output_filename, test_flag)

    # Normalization
    max_x = norm[1]
    min_x = norm[0]
    d = (max_x - min_x)
    indices = [i for i, x in enumerate(d) if x == 0]
    d[indices] = 1
    Z = (X - min_x) / d
    Z[:, 0] = X[:, 0]

    output = predict(Z,
                     filename=model_save)

    mse = ((Y * 255 - output * 255) ** 2).mean(axis=None)

    print('Prediction Completed with a MSE of :' + str(mse))
    if reconstruct_image == 1:
        # display(X_, Y_, name='train')
        display(Z, output, name='out_'+model_save+'_mse_'+str(int(mse)))
    return

def display(X, Y, name):
    Y_channel = X[:, 0]
    # Y_channel = np.ones((Y_channel.shape)) * .5
    num_samples = int(len(Y_channel) / (224 * 224))
    Y_channel = Y_channel.reshape(num_samples, 224, 224, 1)

    CrCb_out = Y.reshape(num_samples, 224, 224, 2)

    out_imgs = np.concatenate((Y_channel, CrCb_out), axis=3).astype(np.float32)
    for i in range(num_samples):
        output_img = (cv2.cvtColor(out_imgs[i, :, :, :], cv2.COLOR_YCR_CB2BGR))
        # cv2.imshow('disp',output_img)

        if name != 'train':
            gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_folder + "gray_"+ str(i) + ".png", gray * 255)

        cv2.imwrite(output_folder + name + '_'+ str(i) + ".png", output_img * 255)
    print(name + ": Images saved at: " + output_folder)
    return

def divide_data(X, Y, num_train, num_val):
    num_train_images = num_train
    total_train = num_train_images * 224 * 224
    total_val = total_train + int(num_val) * 224 * 224  # int((20 - num_train_images) / 2) * 224 * 224

    X_train = X[0:total_train, :]
    Y_train = Y[0:total_train, :]

    # X_train, Y_train = shuffle(X_train, Y_train, random_state=1)

    X_val = X[total_train:total_val, :]
    Y_val = Y[total_train:total_val, :]

    X_test = X[total_val:, :]
    Y_test = Y[total_val:, :]

    return [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]

if test_flag:
    predictor(load_from_file=0)

else:
    train(load_from_file=0)#load "data" [maps +images] from file