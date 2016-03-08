from DBN_linear import test_DBN, predict
import numpy as np
import matplotlib.pyplot as plt
import h5py

train   = 0
prediction = 1

# h5f = h5py.File('data.h5','r')
# b = h5f['dataset'][:]
# h5f.close()

if train == 1:

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

    data = [ (X_train,y_train) , (X_val,y_val),(X_test,y_test)]

    test_DBN(finetune_lr=0.05,  pretraining_epochs=5,   L1_reg=0.00,        k=1,
             pretrain_lr=0.01,  training_epochs=1000,   L2_reg=0.0001,

             dataset=data,      batch_size = 500,
                                layer_sizes=[100,100],
                                output_classes=1)
if prediction == 1:
    output=predict(filename='best_model.pkl')
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