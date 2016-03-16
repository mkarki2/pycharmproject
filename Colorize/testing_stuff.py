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