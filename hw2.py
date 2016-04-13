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
from pprint import pprint
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
    print("Running SGD OLS with lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    np.random.seed(78)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear'))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    #print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    #print("Time:",time()-time0)
    return time()-time0, score

#Problem 1: Write analogous functions
def timed_sgd_Ridge(x1,y1,x2=None,y2=None,
                    lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD Ridge with lambda2: ", lambda2, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    np.random.seed(78)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l2(lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score

def timed_sgd_LASSO(x1,y1,x2=None,y2=None,
            lambda1=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD LASSO with lambda1: ", lambda1, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    np.random.seed(78)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1(lambda1)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True,callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score

#timed_sgd_LASSO(x1,y1,x2,y2)

def timed_sgd_Elastic(x1,y1,x2=None,y2=None,
    lambda1=0.01, lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD Elastic with lambda1: ", lambda1, " lambda2: ", lambda2, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    np.random.seed(78)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1l2(lambda1,lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    print("Score:",score,'\n',np.around(model.layers[0].get_weights()[0][:,0],2))
    print("Time:",time()-time0)
    return time()-time0, score


runOLS = True
runRidge = True
runLASSO = True
runElastic = True


if(runOLS):
    
    best_lr = 0.1
    best_decay = 0.01
    best_mom = 0.8
    best_size = 100
    best_lambda1 = 0.01
    best_lambda2 = 0.01
    best_score = 999999
    best_tt = 999999

    with open("OLS_out.txt", "w") as fout:
        
        for lr in np.arange(0.05,0.5, 0.05):
            for decay in (0.01*x for x in np.arange(0.5, 10.5, 0.5)):
                tt, score = timed_sgd_OLS(x1,y1,x2,y2, lr = lr, decay=decay, momentum=best_mom, batch_size=best_size)
                if score < best_score:
                    best_lr = lr
                    best_decay = decay
                    best_score = score
                    best_tt = tt
                print((best_size,lr,decay,best_mom,tt,score), file=fout)
            fout.flush()    
                
                
             
        print( "Best Score: ", best_score, " Best lr: ", best_lr, "Best decay: ", best_decay)
        
        for mom in np.arange(0.01,0.95, 0.01):
            tt, score = timed_sgd_OLS(x1,y1,x2,y2, lr = best_lr, decay=best_decay, momentum=mom, batch_size=best_size)
            if score < best_score:
                best_mom = mom
                best_score = score
                best_tt = tt
                print((best_size,best_lr,best_decay,mom,tt,score), file=fout)
            fout.flush()  
        print( "Best Score: ", best_score, " Best Momentum: ", best_mom )
        
        for bs in range(50, 510, 10 ):
            tt, score = timed_sgd_OLS(x1,y1,x2,y2, lr = best_lr, decay=best_decay, momentum=best_mom, batch_size=bs)
            if score < best_score:
                best_size = bs
                best_score = score
            print((bs,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()   
        print( "Best Score: ", best_score, " Best size: ", best_size)
    
        print((best_lr, best_decay, best_mom, best_size, best_tt, best_score), file=fout)
          

if(runRidge):
    
    
    best_lr = 0.1
    best_decay = 0.01
    best_mom = 0.8
    best_size = 100
    best_lambda1 = 0.01
    best_lambda2 = 0.01
    best_score = 999999
    best_time = 999999
    best_tt = 999999
    
    
    with open("Ridge_out.txt", "w") as fout:
        
        for lambda2 in np.arange(0.005,0.055, 0.005):
            tt, score = timed_sgd_Ridge(x1,y1,x2,y2, lr = best_lr, lambda2 = lambda2, decay=best_decay, momentum=best_mom, batch_size=best_size)
            if score < best_score:
                best_lambda2 = lambda2
                best_score = score
                best_tt = tt
            print((lambda2,best_size,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()  
        
        for lr in np.arange(0.05,0.5, 0.05):
            for decay in (0.01*x for x in np.arange(0.5, 10.5, 0.5)):
                tt, score = timed_sgd_Ridge(x1,y1,x2,y2, lr = lr, lambda2 = best_lambda2, decay=decay, momentum=best_mom, batch_size=best_size)
                if score < best_score:
                    best_lr = lr
                    best_decay = decay
                    best_score = score
                    best_tt = tt
                print((best_lambda2,best_size,lr,decay,best_mom,tt,score), file=fout)
            fout.flush()    
                
                
             
        print( "Best Score: ", best_score, " Best lr: ", best_lr, "Best decay: ", best_decay)
        
        for mom in np.arange(0.01,0.95, 0.01):
            tt, score = timed_sgd_Ridge(x1,y1,x2,y2, lr = best_lr, lambda2 = best_lambda2, decay=best_decay, momentum=mom, batch_size=best_size)
            if score < best_score:
                best_mom = mom
                best_score = score
                best_tt = tt
            print((best_lambda2,best_size,best_lr,best_decay,mom,tt,score), file=fout)
            fout.flush()  
        print( "Best Score: ", best_score, " Best Momentum: ", best_mom )
        
        
        for bs in range(50, 510, 10 ):
            tt, score = timed_sgd_Ridge(x1,y1,x2,y2, lr = best_lr, lambda2 = best_lambda2, decay=best_decay, momentum=best_mom, batch_size=bs)
            if score < best_score:
                best_size = bs
                best_score = score
                best_tt = tt
            print((best_lambda2,bs,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()
        print( "Best Score: ", best_score, " Best size: ", best_size)
    
        print((best_lambda2, best_lr, best_decay, best_mom, best_size, best_tt, best_score), file=fout)



if(runLASSO):
    
    best_lr = 0.1
    best_decay = 0.01
    best_mom = 0.8
    best_size = 100
    best_lambda1 = 0.01
    best_lambda2 = 0.01
    best_score = 999999
    best_tt = 999999
    
    
    with open("LASSO_out.txt", "w") as fout:
        
        for lambda1 in np.arange(0.005,0.055, 0.005):
            tt, score = timed_sgd_LASSO(x1,y1,x2,y2, lr = best_lr, lambda1 = lambda1, decay=best_decay, momentum=best_mom, batch_size=best_size)
            if score < best_score:
                best_lambda1 = lambda1
                best_score = score
                best_tt = tt
            print((lambda1,best_size,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()  
        
        for lr in np.arange(0.05,0.5, 0.05):
            for decay in (0.01*x for x in np.arange(0.5, 10.5, 0.5)):
                tt, score = timed_sgd_LASSO(x1,y1,x2,y2, lr = lr, lambda1 = best_lambda1, decay=decay, momentum=best_mom, batch_size=best_size)
                if score < best_score:
                    best_lr = lr
                    best_decay = decay
                    best_score = score
                print((best_lambda1,best_size,lr,decay,best_mom,tt,score), file=fout)
            fout.flush()    
                
                
             
        print( "Best Score: ", best_score, " Best lr: ", best_lr, "Best decay: ", best_decay)
        
        for mom in np.arange(0.01,0.95, 0.01):
            tt, score = timed_sgd_LASSO(x1,y1,x2,y2, lr = best_lr, lambda1 = best_lambda1, decay=best_decay, momentum=mom, batch_size=best_size)
            if score < best_score:
                best_mom = mom
                best_score = score
                best_tt = tt
            print((best_lambda1,best_size,best_lr,best_decay,mom,tt,score), file=fout)
            fout.flush()  
        print( "Best Score: ", best_score, " Best Momentum: ", best_mom )
        
        
        for bs in range(50, 510, 10 ):
            tt, score = timed_sgd_LASSO(x1,y1,x2,y2, lr = best_lr, lambda1 = best_lambda1, decay=best_decay, momentum=best_mom, batch_size=bs)
            if score < best_score:
                best_size = bs
                best_score = score
            print((best_lambda1,bs,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()
        print( "Best Score: ", best_score, " Best size: ", best_size)
    
        print((best_lambda1, best_lr, best_decay, best_mom, best_size, best_tt, best_score), file=fout)


if(runElastic):
    
    best_lr = 0.1
    best_decay = 0.01
    best_mom = 0.8
    best_size = 100
    best_lambda1 = 0.01
    best_lambda2 = 0.01
    best_score = 999999
    best_tt = 999999
    
    
    with open("Elastic_out.txt", "w") as fout:
        
        for lambda1 in np.arange(0.005,0.055, 0.005):
            for lambda2 in np.arange(0.005,0.055, 0.005):
                tt, score = timed_sgd_Elastic(x1,y1,x2,y2, lr = best_lr, lambda1 = lambda1, lambda2 = lambda2, decay=best_decay, momentum=best_mom, batch_size=best_size)
                if score < best_score:
                    best_lambda2 = lambda2
                    best_lambda1 = lambda1
                    best_score = score
                    best_tt = tt
                print((lambda1, lambda2, best_size,best_lr,best_decay,best_mom,tt,score), file=fout)
                fout.flush()    
        
        for lr in np.arange(0.05,0.5, 0.05):
            for decay in (0.01*x for x in np.arange(0.5, 10.5, 0.5)):
                tt, score = timed_sgd_Elastic(x1,y1,x2,y2, lr = lr, lambda1 = best_lambda1, lambda2 = best_lambda2, decay=decay, momentum=best_mom, batch_size=best_size)
                if score < best_score:
                    best_lr = lr
                    best_decay = decay
                    best_score = score
                    best_tt = tt
                print((best_lambda1,best_size,lr,decay,best_mom,tt,score), file=fout)
            fout.flush()    
                
                
             
        print( "Best Score: ", best_score, " Best lr: ", best_lr, "Best decay: ", best_decay)
        
        for mom in np.arange(0.01,0.95, 0.01):
            tt, score = timed_sgd_Elastic(x1,y1,x2,y2, lr = best_lr, lambda1 = best_lambda1, lambda2 = best_lambda2, decay=best_decay, momentum=mom, batch_size=best_size)
            if score < best_score:
                best_mom = mom
                best_score = score
                best_tt = tt
            print((best_lambda1,best_lambda2,best_size,best_lr,best_decay,mom,tt,score), file=fout)
            fout.flush()  
        print( "Best Score: ", best_score, " Best Momentum: ", best_mom )
           
        
        for bs in range(50, 510, 10 ):
            tt, score = timed_sgd_Elastic(x1,y1,x2,y2, lr = best_lr, lambda1 = best_lambda1, lambda2 = best_lambda2, decay=best_decay, momentum=best_mom, batch_size=bs)
            if score < best_score:
                best_size = bs
                best_score = score
                best_tt = tt
            print((best_lambda1,best_lambda2,bs,best_lr,best_decay,best_mom,tt,score), file=fout)
            fout.flush()
        print( "Best Score: ", best_score, " Best size: ", best_size)
    
        print((best_lambda1, best_lambda2, best_lr, best_decay, best_mom, best_size, best_tt, best_score), file=fout)

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