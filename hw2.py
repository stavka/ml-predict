# Problem 0(Not graded): install keras, TensorFlow
# Instruction can be found at:
# http://keras.io/#installation
# https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation
# then add to your .bashrc
# export KERAS_BACKEND=tensorflow
# (Hint: this is all linux, if you have windows, then linux vm or see docker installation but it would be easier to run on linux.)

# import keras
from keras.models import Sequential
from keras.objectives import mae
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l1,l2,l1l2
from keras.callbacks import EarlyStopping
from time import time
import numpy as np
import pprint
import scipy.optimize as scipy_opt
from math import isnan

N,K = 1000,10
np.random.seed(78) #Make sure to use seed 78 for grading
def generate(N,K): #lazy generator
    res = np.random.randn(N,K); #make N 
    res[:,0]=1; # replace first variable with constant
    return res
f = (lambda x: x.dot(np.arange(K)-1))
noise_multiplier = 1
x1 = generate(4*N,K)
y1 = f(x1) + noise_multiplier * K * np.random.randn(4*N)
x2 = generate(N,K)
y2 = f(x2) + noise_multiplier * K * np.random.randn(N)
x3 = generate(N,K)
y3 = f(x3) + noise_multiplier * K * np.random.randn(N)

def timed_sgd_OLS(x1,y1,x2=None,y2=None,
                  lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1, verbose=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear'))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper])
    score = model.evaluate(x2, y2, batch_size=20)
    #print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    #print("Time:",time()-time0)
    return time()-time0, score



results = []

for bs in (10, 50, 100, 200, 500):
    for lr in (0.001,0.01,0.02,0.05,0.1, 0.2, 0.5, 0.75):
        for decay in (1e-2*x for x in (0.01, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50.)):
            for mom in (0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95):
        
                results.append((bs,lr,decay,mom,timed_sgd_OLS(x1,y1,x2,y2, lr = lr, decay=decay, momentum=mom, batch_size=bs)))
 
pprint.pprint(results)       


# results = []
#  
# for lr in (0.001,0.01,0.02,0.05,0.1, 0.2, 0.5, 0.75):
#     results.append((lr,timed_sgd_OLS(x1,y1,x2,y2, lr = lr)))
#      
# pprint.pprint(results)

#best_lr = 0.01

#results = []

#for decay in (1e-2*x for x in (0.01, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50.)):
#    results.append((decay,timed_sgd_OLS(x1,y1,x2,y2, lr = best_lr, decay=decay)))

#pprint.pprint(results)


def training_function( x, x1, y1, x2, y2 ):
    lr, decay, momentum, batch_size = x[0],x[1],x[2], 100
    tt, score = timed_sgd_OLS(x1=x1,y1=y1,x2=x2,y2=y2, lr=lr, decay=decay, nesterov=True, momentum=momentum, batch_size=batch_size )
    if isnan(score):
            score = 1000 
    return tt*tt+score*score
 
 
#(0.1, 1e-2, 0.8, 100)
 
 
# min_result = scipy_opt.minimize( training_function, (0.01, 5e-2, 0.8, 100), args=(x1,y1,x2,y2),
#                                   method='L-BFGS-B', jac=None,
#                                 bounds=((0.01,0.5),(1e-4,0.1),(0.1,0.95),(10, 200)) )


min_result = scipy_opt.basinhopping( training_function, (0.01, 5e-2, 0.8), minimizer_kwargs = { 'args' : (x1,y1,x2,y2) } )

  
pprint.pprint(min_result)

#x': array([  1.00000030e-01,   1.00000068e-02,   8.00000011e-01,
#         1.00000008e+02])}

# results = timed_sgd_OLS(x1=x1,y1=y1,x2=x2,y2=y2, lr=1.00000030e-01, decay=1.00000068e-02, nesterov=True, momentum=8.00000011e-01, batch_size=100 )
# 
# pprint.pprint(results)


#Problem 1: Write analogous functions
def timed_sgd_Ridge(x1,y1,x2=None,y2=None,
                    lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    time0 = time() # Start timer
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l2(lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score




#timed_sgd_Ridge(x1,y1,x2,y2)

def timed_sgd_LASSO(x1,y1,x2=None,y2=None,
            lambda1=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    time0 = time() # Start timer
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1(lambda1)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score

#timed_sgd_LASSO(x1,y1,x2,y2)

def timed_sgd_Elastic(x1,y1,x2=None,y2=None,
    lambda1=0.01, lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    time0 = time() # Start timer
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1l2(lambda1,lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score

#timed_sgd_Elastic(x1,y1,x2,y2)

#Problem 2: 
#Part A: Your goal here is to obtain better time and score than default parameters for each of the four above functions:
#timed_sgd_OLS, timed_sgd_Ridge, timed_sgd_LASSO, timed_sgd_Elastic when run with seed(78) and x1,y1,x2,y2 as defined above.
#Part B: Same but with random seed instead of seed(78); if you did Part A right, this part
#is trivial but you should be able to obtain better performance in almost every case.
#You answer should show code by which you found these parameters with some brief explanation.

#Suggested path: for each function using this seed(78) while keeping other parameters the same, find better lambda2,
#then repeat same for lr, decay, nesterov, momentum, batch_size while each time updating previous parameters to newly
#found better parameters.
#For example you found that lambda2 is better to be 0.1,
#then you optimize next lr while setting lambda to 0.1 and rest of parameters the same and if lr=0.01
#is better, etc.  You can choose any other path 
#(EXTRA CREDIT WILL BE GIVEN AT DISCRETION FOR: showing application of any techniques from class to this problem; 
# obtaining results significantly better than median of the class; the goal is to be student with best accuracy 
# and reasonable time)

#Problem 3: From running problem 2, make observation of what happens when you change each of the parameters:
#a) Learning_rate (i.e. lr)
#b) decay
#c) momentum
#d) Nesterov
#e) batch_size
#f) nb_epoch
#one line explanation for each case suffices