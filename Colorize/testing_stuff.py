import numpy as np
import matplotlib.pyplot as plt
import h5py
from my_utility import tic, toc
from sklearn.utils import shuffle
import cv2
train = 1
prediction = 0

tic()
f = h5py.File('data_YCrCb_normalized.h5', 'r')
X = f['/data/X'][:]
Y = f['/data/Y'][:]
norm_y= f['/data/norm_y'][:]
# norm_c =f['/data/norm_c'][:]
f.close()
toc("Data loaded from file.")

tmp=X[:,0]
tmp2= tmp.reshape(20,224,224)
tmp3= tmp2[0,:,:]
norm_val = norm_y[:,0]
tmp4=tmp3*norm_val[1]
tmp4=tmp4+norm_val[0]

Y_channel= tmp4

tmp=Y[:,0]
tmp2= tmp.reshape(20,224,224)
tmp3= tmp2[0,:,:]
# norm_val = norm_c[0]
# tmp4=tmp3*norm_val[1]
# tmp4=tmp4+norm_val[0]
Cr=tmp3

tmp=Y[:,1]
tmp2= tmp.reshape(20,224,224)
tmp3= tmp2[0,:,:]
# norm_val = norm_c[1]
# tmp4=tmp3*norm_val[1]
# tmp4=tmp4+norm_val[0]
Cb=tmp3
YCrCb=np.zeros((224,224,3))

YCrCb[:,:,0]=Y_channel
YCrCb[:,:,1]=Cr
YCrCb[:,:,2]=Cb

img = np.array(YCrCb, dtype=np.float32)
im_converted = (cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR))

cv2.imshow('',im_converted)
print('Done')


# im_original = cv2.resize(cv2.imread('/home/exx/MyTests/MATLABTests/val_images/ILSVRC2012_val_00000001.JPEG'), (224, 224)).astype(np.float32)/255
# img = (cv2.cvtColor(im_original, cv2.COLOR_BGR2YCR_CB))
# #3
# #...
# #n-2
# im_converted = (cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR))
# cv2.imshow('',im_original)
# t=1

first_imgs=X[:,0]
first_imgs=first_imgs.reshape(30,224,224)
first_img=first_imgs[0,:,:]
Y_=first_img.reshape(224,224).astype(np.float32)
Y_=np.expand_dims(Y_,axis=2)

# RB=YCrCb[0, :, :, 1:].astype(np.float32)
RBs=Y.reshape(30,224,224,2)

RB=RBs[0,:,:,:]
RB=RB.reshape(224,224,2)
YRB=np.concatenate((Y_,RB),axis=2).astype(np.float32)


output_img = (cv2.cvtColor(YRB, cv2.COLOR_YCR_CB2BGR))
cv2.imshow('disp',output_img)
output_folder = "/home/exx/PycharmProjects/Output_Imgs/"
cv2.imwrite(output_folder + "temp_" + str(1) + ".jpg", output_img*255)