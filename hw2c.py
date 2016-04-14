from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.regularizers import l1,l2,l1l2
from keras.callbacks import EarlyStopping
from time import time
import numpy as np
from collections import OrderedDict

N,K = 1000,10

rseed = 78  #Make sure to use seed 78 for grading

def generate(N,K): #lazy generator
    res = np.random.randn(N,K); #make N 
    res[:,0]=1; # replace first variable with constant
    return res
f = (lambda x: x.dot(np.arange(K)-1))
noise_multiplier = 1
np.random.seed(rseed)
x1 = generate(4*N,K)
y1 = f(x1) + noise_multiplier * K * np.random.randn(4*N)
x2 = generate(N,K)
y2 = f(x2) + noise_multiplier * K * np.random.randn(N)
x3 = generate(N,K)
y3 = f(x3) + noise_multiplier * K * np.random.randn(N)

def timed_sgd_OLS(x1,y1,x2=None,y2=None,
                  lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(y2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD OLS with lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear'))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    return time()-time0, score


#Problem 1: Write analogous functions
def timed_sgd_Ridge(x1,y1,x2=None,y2=None,
                    lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(x2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD Ridge with lambda2: ", lambda2, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l2(lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    return time()-time0, score

def timed_sgd_LASSO(x1,y1,x2=None,y2=None,
            lambda1=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(y2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD LASSO with lambda1: ", lambda1, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1(lambda1)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True,callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    return time()-time0, score


def timed_sgd_Elastic(x1,y1,x2=None,y2=None,
    lambda1=0.01, lambda2=0.01, lr=0.1, decay=1e-2, nesterov=True, momentum=0.8, batch_size=100,nb_epoch=50):
    if (x2 is None)^(y2 is None): raise ValueError("if you specify x2 or y2 you need to specify the other as well")
    if x2 is None and y2 is None: x2=x1; y2=y1 #if no Cross-validation set, use original
    print("Running SGD Elastic with lambda1: ", lambda1, " lambda2: ", lambda2, " lr: ", lr, " decay: ", decay, " Momentum: ", momentum, " Batch size: ", batch_size)
    time0 = time() # Start timer
    earlystopper = EarlyStopping(monitor='loss', patience=1)
    sgd=SGD(lr=lr, decay=decay, nesterov=nesterov, momentum=momentum)
    model = Sequential()
    model.add(Dense(1, input_dim=K, activation='linear', W_regularizer=l1l2(lambda1,lambda2)))
    model.compile(loss='mse', optimizer=sgd)
    model.fit(x1, y1,nb_epoch=nb_epoch,batch_size=batch_size, show_accuracy=True, callbacks=[earlystopper], verbose=0)
    score = model.evaluate(x2, y2, batch_size=20)
    return time()-time0, score


def tuneParameters( objfun, params, start_params, fout=None):
    
    threshold_for_imp = 1.2     ## threshold fo time improvement
    
    best_params = { p:start_params[p] for p in params.keys() }
    best_score = 999999
    best_tt = 999999
    
    for param in params.keys():
        for pv in np.arange(params[param][0], params[param][1], params[param][2]):
            start_params[param] = pv
            np.random.seed(rseed)
            tt, score = objfun(**start_params)
            if (score < best_score) & (tt < best_tt * threshold_for_imp):
                best_score = score
                best_tt = tt
                best_params[param] = pv
                start_params[param] = pv
            if fout != None:
                print( (score, tt, { p: start_params[p] for p in params.keys() }), file=fout)
        start_params[param] = best_params[param]
        
    if fout != None:    
        print( (best_score, best_tt, best_params), file=fout)
        
    return best_score, best_tt, best_params

runOLS = True
runRidge = True
runLASSO = True
runElastic = True


if(runOLS):
    
    params = OrderedDict()
    params['batch_size'] = (50, 550, 50 )
    params['momentum'] = (0,0.95, 0.05)
    params['decay'] = (0.005, 0.105, 0.005)
    params['lr'] = (0.05,0.5, 0.05)
   
    
    start_params = { 'x1' : x1,
                     'y1' : y1,
                     'x2' : x2,
                     'y2' : y2,
                     'lr' : 0.1,
                     'decay': 0.01,
                     'momentum': 0.8,
                     'batch_size': 100  }
    

    with open("OLS_out.txt", "w") as fout:
        tuneParameters( timed_sgd_OLS, params, start_params, fout=fout)
                

if(runRidge):

    
    params = OrderedDict()
    params['batch_size'] = (50, 550, 50 )
    params['lambda2'] = (0.005,0.055, 0.005)
    params['momentum'] = (0,0.95, 0.05)
    params['decay'] = (0.005, 0.105, 0.005)
    params['lr'] = (0.05,0.5, 0.05)
    
   
    
    start_params = { 'x1' : x1,
                     'y1' : y1,
                     'x2' : x2,
                     'y2' : y2,
                     'lr' : 0.1,
                     'decay': 0.01,
                     'momentum': 0.8,
                     'batch_size': 100,
                     'lambda2': 0.01  }
    
    
    with open("Ridge_out.txt", "w") as fout:
        tuneParameters( timed_sgd_Ridge, params, start_params, fout=fout)
        

if(runLASSO):
    
    params = OrderedDict()
    params['batch_size'] = (50, 550, 50 )
    params['lambda1'] = (0.005,0.055, 0.005)
    params['momentum'] = (0,0.95, 0.05)
    params['decay'] = (0.005, 0.105, 0.005)
    params['lr'] = (0.05,0.5, 0.05)
    
   
    
    start_params = { 'x1' : x1,
                     'y1' : y1,
                     'x2' : x2,
                     'y2' : y2,
                     'lr' : 0.1,
                     'decay': 0.01,
                     'momentum': 0.8,
                     'batch_size': 100,
                     'lambda1': 0.01  }
    
    
    with open("LASSO_out.txt", "w") as fout:
        tuneParameters( timed_sgd_LASSO, params, start_params, fout=fout)        
    


if(runElastic):
    
    params = OrderedDict()
    params['batch_size'] = (50, 550, 50 )
    params['lambda1'] = (0.005,0.055, 0.005)
    params['lambda2'] = (0.005,0.055, 0.005)
    params['momentum'] = (0,0.95, 0.05)
    params['decay'] = (0.005, 0.105, 0.005)
    params['lr'] = (0.05,0.5, 0.05)
    
   
    
    start_params = { 'x1' : x1,
                     'y1' : y1,
                     'x2' : x2,
                     'y2' : y2,
                     'lr' : 0.1,
                     'decay': 0.01,
                     'momentum': 0.8,
                     'batch_size': 100,
                     'lambda1': 0.01,
                     'lambda2': 0.01  }
    
    
    with open("Elastic_out.txt", "w") as fout:
        tuneParameters( timed_sgd_Elastic, params, start_params, fout=fout)











