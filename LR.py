#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn.preprocessing import MinMaxScaler
#x = np.array([[1, 2, 3], [1,1,1],[1,2,1]])
#y = np.array([1,3,2])
#w = np.array([0,0,0], dtype='f')

def get_features():
    #import data frame into pandas from csv
    df=pd.read_csv(r"../input/filles/t.csv")

    #add extra column for 1 in dataframe
    df.insert(0,'new',1)
    #dataframe to numpy array calculator
    x=df.to_numpy()
    x=np.vstack(x)

    
    #delete entry number of data that was in earlier csv
    x=np.delete(x,1,1)
    
    #deleting MODIS column
    x=np.delete(x,9,1)
    #deleting 6.0NRT column
    x=np.delete(x,10,1)
    
    
    #declaring numpy array for w
    p=x.shape[0]
    
    
    y=np.zeros(p, dtype = 'f')

    p=x.shape[0]
   
    count=0

    for i in range(0,25001):
        if x[i][8]=='Terra':
               x[i][8]=1
        if x[i][8] == 'Aqua':
               x[i][8]=1.1
        if x[i][12]=='D':
               x[i][12]=1
        if x[i][12]=='N':  
               x[i][12]=1.01
        x[i][6] = 1
        
        y[i]=x[i][11]
    
    x=np.delete(x,11,1)
    #del longitude
    #x=np.delete(x,2,1)
    #del latitude
    #x=np.delete(x,1,1)
    sc = MinMaxScaler()
    #nx = sc.fit_transform(x)
    for i in range(0,25001):
        x[i][0]=1
    #nx = preprocessing.scale(x)
    #nx = preprocessing.normalize(x)
    l1=len(x[0])
    x[:,1:2] = sc.fit_transform(x[:,1:2])
    x[:,2:3] = sc.fit_transform(x[:,2:3])
    x[:,3:4] = preprocessing.scale(x[:,3:4])
    x[:,4:5] = preprocessing.scale(x[:,4:5])
    x[:,6:7] = sc.fit_transform(x[:,6:7])
    x[:,7:8] = sc.fit_transform(x[:,7:8])
    x[:,10:11] = preprocessing.scale(x[:,10:11])
    x[:,9:10] = preprocessing.scale(x[:,9:10])
    
    
    w=np.zeros(l1, dtype = 'f')
    
    
    
    return x,y,w
###############


def dev_get_features():
    #import data frame into pandas from csv
    df=pd.read_csv(r"../input/filles/d.csv")

    #add extra column for 1 in dataframe
    df.insert(0,'new',1)
    #dataframe to numpy array calculator
    x=df.to_numpy()
    x=np.vstack(x)

    
    #delete entry number of data that was in earlier csv
    x=np.delete(x,1,1)
    
    #deleting MODIS column
    x=np.delete(x,9,1)
    #deleting 6.0NRT column
    x=np.delete(x,10,1)
    
    
    #declaring numpy array for w
    p=x.shape[0]
    
    
    y=np.zeros(p, dtype = 'f')

    p=x.shape[0]
   
    count=0

    for i in range(0,4001):
        if x[i][8]=='Terra':
               x[i][8]=1
        if x[i][8] == 'Aqua':
               x[i][8]=1.1
        if x[i][12]=='D':
               x[i][12]=1
        if x[i][12]=='N':  
               x[i][12]=1.01
        x[i][6] = 1
        
        y[i]=x[i][11]
    
    x=np.delete(x,11,1)
    #del longitude
    #x=np.delete(x,2,1)
    #del latitude
    #x=np.delete(x,1,1)
    sc = MinMaxScaler()
    #nx = sc.fit_transform(x)
    for i in range(0,4001):
        x[i][0]=1
    #nx = preprocessing.scale(x)
    #nx = preprocessing.normalize(x)
    l1=len(x[0])
    x[:,1:2] = sc.fit_transform(x[:,1:2])
    x[:,2:3] = sc.fit_transform(x[:,2:3])
    x[:,3:4] = preprocessing.scale(x[:,3:4])
    
    x[:,4:5] = preprocessing.scale(x[:,4:5])
    x[:,6:7] = sc.fit_transform(x[:,6:7])
    x[:,7:8] = sc.fit_transform(x[:,7:8])
    x[:,10:11] = preprocessing.scale(x[:,10:11])
    x[:,9:10] = preprocessing.scale(x[:,9:10])
    w=np.zeros(l1, dtype = 'f')
    
    
    
    return x,y,w
###############

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''
    mse=0
    n=len(targets)
    for i in range(n):
        sum=0
        for j in range(len(weights)):
            sum+=round((weights[j]*feature_matrix[i][j]),10)
            
        diff=sum-targets[i]
        
        mse+=round((diff)*(diff),10)/len(feature_matrix)
    return mse

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''


def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    l2=0
    for i in weights:
        l2+=(i*i)
    return round(l2,10)
    '''
    Arguments
    weights: numpy array of shape n x 1
    '''


def loss_fn(feature_matrix, weights, targets, C=0.4):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''
    
    total_loss=round(mse_loss(feature_matrix, weights, targets)+(C*l2_regularizer(weights)),10)
    
    #print(total_loss)
    return total_loss
    
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''





def compute_gradients(feature_matrix, weights, targets, C=0.8):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''
    grad = np.zeros(len(weights))
    
    for i in range(len(weights)):
        instance_sum=0.0
        for j in range(len(targets)):
            wTx=0.0
            for k in range(len(weights)):
                    wTx+=round((weights[k]*feature_matrix[j][k]),10)
            diff=(wTx-targets[j])
            diff*=feature_matrix[j][i]
            instance_sum+=round(diff,10)
        #instance_sum*=2 #check whether sigma needed
        instance_sum/=len(feature_matrix)
        l2=C*weights[i]  # C represents 2*C
        grad[i]=round(instance_sum,10)+l2
        
    return grad
        
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    
def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''

    return weights
 
def basis1(x):
    p=x.shape[0]
    for i in range(0,p-1):
        x[i][3]=x[i][3]**2+x[i][3]+2
        x[i][4]=(2*x[i][4]**2)+x[i][4]+2
        x[i][5]=(x[i][5]**2)+x[i][5]+x[i][5]+3
        if x[i][7]>200 and x[i][7]<800:
            x[i][7]=(x[i][7]**2)+x[i][7]+2
        else :
            x[i][7]=x[i][7]/4
        x[i][8]=x[i][8]*x[i][8]+2
        x[i][9]=x[i][9]**2
    return x
    
     
def basis2(x):
    p=x.shape[0]
    for i in range(0,p-1):
        x[i][3]=x[i][3]**2+(1)/(1+pow(2.303,-x[i][3]))
        x[i][4]=x[i][4]**2+(2)/(1+pow(2.303,-x[i][4]))
        x[i][5]=x[i][5]**2+(3)/(1+pow(2.303,-x[i][5]))
        x[i][6]=x[i][6]**2+(2)/(1+pow(2.303,-x[i][6]))
        x[i][7]=x[i][7]**2+(2)/(1+pow(2.303,-x[i][7]))
        x[i][8]=x[i][8]**2+(3)/(1+pow(2.303,-x[i][8]))
        x[i][9]=x[i][9]**2+(1)/(1+pow(2.303,-x[i][9]))
    return x
    

    
  
    
def fun():
    x,y,w=get_features()
    dx,dy,dw=dev_get_features()
    x=basis1(x)
    dx=basis1(dx)
    print(x.shape[1])
    #x=x[:500]
    #y=y[:500]
    #wt=[-467108442708153.94,-0.07861620845904875,-0.07975692231124644,5.155555,156.6942,-300.086596,467108442726896.5,0.041176206310853926,-5.57,-0.79263,-0.624458,9.436905]
    #print(round(loss_fn(x,wt,y,0.5),10))
    print(loss_fn(x,w,y,0.5))
    print(w)
    lr=0.0000008
    k=0
    
    lf_old=math.inf
    lf=0
    
    while(k<25001):
        while(True):
            nx=x[k:k+1000,]
            ny=y[k:k+1000]
            grad=compute_gradients(nx, w, ny, 0.5)
            #print(grad)
            for j in range(len(w)):
                w[j]-=round((0.002*grad[j]),10)
            lf=round(loss_fn(nx,w,ny,0.5),10)
            print(w)
            print("train error:  ",lf)
            print("dev error:  ",round(loss_fn(dx,w,dy,0.5),10))
            
            if(abs((round(lf_old,3))-(round(lf,3))) <=1 ):
                break
            lf_old=lf
        k+=1000
        print("#############  Batch  ################")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
""" 
    while(k<25001):
        for i in range(k,k+1000,1):
            nx=x[i:i+1000,]
            ny=y[i:i+1000]
            grad=compute_gradients(nx, w, ny, 0.5)
            #print(grad)
            for j in range(len(w)):
                w[j]-=round((0.002*grad[j]),10)
            lf=round(loss_fn(nx,w,ny,0.5),10)
            print(w)
            print("train error:  ",lf)
            print("dev error:  ",round(loss_fn(dx,w,dy,0.5),10))
            
            if(abs((round(lf_old,3))-(round(lf,3))) <=1 ):
                break
            lf_old=lf
            
        k=k+1000
        print("#############  Batch  ################")
"""

#grad=compute_gradients(x, w, y, 0.5)  
#print(grad)

  
fun()

