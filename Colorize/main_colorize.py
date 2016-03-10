from matplotlib import pyplot as plt

import theano
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import cv2
import os
import h5py
from my_utility import tic, toc
import pickle

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 1
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 3
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 6
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 8
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 11
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 13
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 15
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 18
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 20
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 22
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 25
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 27
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 29
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def Convert2YCrCb(folder,num_samples):
    tic()
    image_list = os.listdir(folder)
    YCrCb = np.zeros((num_samples,3,224,224))
    for i in range(len(image_list)):
        if image_list[i].endswith(".JPEG"):
            im_original = cv2.resize(cv2.imread(folder + image_list[i]), (224, 224)).astype(np.float32)
            im_original[:, :, 0] -= 103.939
            im_original[:, :, 1] -= 116.779
            im_original[:, :, 2] -= 123.68

            # Converting to YCrCb
            im_converted =(cv2.cvtColor(im_original, cv2.COLOR_BGR2YCR_CB))
            #im_converted[:, :, 0] -= 97.7
            # im_converted[:, :, 1] -= 5.436
            # im_converted[:, :, 2] -= -5.63

            im = im_converted.transpose((2, 0, 1))

            im = np.expand_dims(im, axis=0)

            YCrCb[i]=im
        if (i+1)%num_samples==0:
                break

    toc("Images Converted to YCrCb.")
    return YCrCb

def GenerateMaps(model, Y_Images):
    tic()
    num_samples= len(Y_Images)

    layers_extract = [3, 8, 15, 22, 29]

    all_hc = np.zeros((num_samples, 1473, 50176))

    layers = [model.layers[li].get_output(train=False) for li in layers_extract]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)

    def extract_hypercolumn(instance):
        # dicti=[model.layers[li].get_config() for li in layer_indexes]
        feature_maps = get_feature(instance)
        hypercolumns = np.zeros((1473,50176))
        hypercolumns[0] = np.reshape(instance[:,0,:,:], (1, 50176))
        ctr = 1
        for convmap in feature_maps:
            for fmap in convmap[0]:
                upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                            mode="F", interp='bilinear')

                hypercolumns[ctr] = np.reshape(upscaled, (1, 50176))
                ctr += 1
        return np.asarray(hypercolumns)

    print("Starting Loop")
    counter = 0
    for i in range(len(Y_Images)):

        Y_Channel = np.expand_dims(Y_Images[i, :, :], axis=0)
        Y_Image = np.append(Y_Channel,Y_Channel,axis=0)
        Y_Image = np.append(Y_Image,Y_Channel,axis=0)
        Y_Image = np.expand_dims(Y_Image, axis=0).astype(np.float32)

        hc = extract_hypercolumn(Y_Image)
        hc = np.expand_dims(hc, axis=0)

        all_hc[counter] = hc

        counter += 1
        if not counter % 5:
            print(counter)
        if not counter % num_samples:
            break

    toc("Hypercolumns Extracted.")
    return all_hc

def load_model(filename,weights):

    tic()
    if weights==0: #    1: load from weights file, 0: load from pickle file
        model = pickle.load(open ('kerasmodel','rb'))
    else:
        model = VGG_16(filename)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

    toc("Model Loaded. Compiled.")
    #pickle.dump(model,open ('kerasmodel','wb'))
    return model

def save_data(X,Y,output_filename):
    tic()
    f = h5py.File(output_filename, 'w')
    grp=f.create_group("data")
    grp.create_dataset('X', data=X, compression="gzip")
    grp.create_dataset('Y', data=Y, compression="gzip")

    f.close()
    toc('Data File saved to disk.')


def CreateTargets(CrCb):
    num_samples = len(CrCb)
    targets=np.reshape(CrCb,(num_samples,2,50176))

    return targets



if __name__ == '__main__':


    model = load_model('/home/exx/vgg16_weights.h5',weights=0)
    folder = '/home/exx/MyTests/MATLABTests/val_images/'

    num_samples=20

    YCrCb = Convert2YCrCb(folder,num_samples)
    maps = GenerateMaps(model, YCrCb[:,0,:,:])
    targets = CreateTargets(YCrCb[:,1:,:,:])
    maps=maps.reshape(num_samples*224*224,1473)
    targets=targets.reshape(num_samples*224*224,2)

    output_filename = 'data_YCrCb.h5'
    save_data(maps,targets,output_filename)


