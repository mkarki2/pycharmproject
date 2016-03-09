import cv2
import main_colorize as hc
import numpy as np
import matplotlib.pyplot as plt


def CreateData(Images):
    I = Convert2YCrCb(Images)  # Regular Images to YCbCr

    Y_Image = []
    Y_Image[:, :, 0, :] = I[:, :, 0, :]
    Y_Image[:, :, 1, :] = I[:, :, 0, :]
    Y_Image[:, :, 2, :] = I[:, :, 0, :]

    X = GenerateMaps(Y_Image[:, :, :, 0])  # For now just 1

    # Making Targets (ie. Cb-Cr Target Vectors)
    Y = CreateTargets(I)

    return X, Y, Y_Image[:, :, 0, :]


# Convert RGB Image(s) to YCbCr Channel
def Convert2YCrCb(Images):
    imgYCbCr = []
    for i in range(len(Images)):
        np.append(imgYCbCr, (cv2.cvtColor(Images[i], cv2.COLOR_BGR2YCR_CB)), axis=0)
    return imgYCbCr


# TODO: Generate Concatenated and properly formatted maps as X
def GenerateMaps(Y_Image):
    maps = hc.extract_hypercolumn(Y_Image)
    # TODO: reshape and resize maps properly
    return maps


# TODO: Create floating point targets (u,v) as Y
def CreateTargets(I):
    Cr = I[:, :, 1, :]
    Cb = I[:, :, 2, :]
    CrCb = [Cr, Cb]
    # TODO: reshape CrCb
    return CrCb


def Colorize(Images, trained_model):
    X, Y, Y_Channel = CreateData(Images)

    # Predict on Pretrained network
    CrCb = Predict(X, trained_model)

    accuracy = CheckAccuracy(CrCb, Y)

    print('Prediction Accuracy: ' + (accuracy * 100) + '%')

    # Displaying Colored Images alongside Black and White Goes Here
    IMG = AddColor(Y_Channel, CrCb)  # output is RGB images

    DisplayImages(IMG, Y_Channel, Images)

    return

def Predict(X, trained_model):
    # TODO:predict values
    CrCb=0
    return CrCb

def CheckAccuracy(CrCb,Y):
    # TODO:calculate accuracy
    accuracy=0
    return accuracy

def AddColor(Y_Channel, CrCb):
    # TODO:Add Color to Y Channel
    IMG=0
    return IMG

def DisplayImages(IMG, Y_Channel, Images):
    # TODO:Display Images (Color and Black and White)
    return


