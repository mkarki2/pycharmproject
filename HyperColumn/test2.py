from neural_network_regression import predict,do_mlp
import numpy as np

predict_flag= 1

if (predict_flag ==0):
    X_train = np.random.rand(5000,500)*.75 +.25
    X_train = np.append(X_train,np.random.rand(5000,500)*.75,axis=0)

    y_train = np.random.rand(5000,)*.3
    y_train = np.append(y_train,np.random.rand(5000,)*.3+.7,axis=0)

    X_val = np.random.rand(1000,500)*.75 +.25
    X_val = np.append(X_val,np.random.rand(1000,500)*.75,axis=0)

    y_val = np.random.rand(1000,)*.3
    y_val = np.append(y_val,np.random.rand(1000,)*.3+.7,axis=0)

    X_test = np.random.rand(1000,500)*.75 +.25
    X_test = np.append(X_test,np.random.rand(1000,500)*.75,axis=0)

    y_test = np.random.rand(1000,)*.3
    y_test = np.append(y_test,np.random.rand(1000,)*.3+.7,axis=0)
    data = [ (X_train,y_train) , (X_val,y_val),(X_test,y_test) ]
    do_mlp(dataset=data)
else:
    predict()