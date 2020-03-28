# python3 Neural Network
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:05:50 2020

@author: mahyar
"""


#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Important Functions
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def relu(Z):
    return np.max(0,Z)
#Initializing Weights using He initialization to overcome gradient exploding
def Initialize_weights(layers_dims):
    '''[He Initialization] Layers_dims is a vector whose length indicates the number of layers
    and the element at each index corresponds to the number of neurons in that layer
    index 0 represents the input layer'''
    np.random.seed(3)
    L = len(layers_dims)
    weights = dict({})
    for l in range(1,L):
        weights['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        weights['b'+str(l)] = np.zeros([layers_dims[l],1])
    return weights

def compute_Activation(W,A_prev,b,activation_func):
    '''Computes the actiavtion based on the type of activation function.
    Required:
        W : Weights
        A_prev: X or previous perceptron
        activation_func: sigmoid or relu'''
    Z = np.dot(W,A_prev)+b
    linear_cache = (A_prev,W,b)
    if(activation_func == 'relu'):
        A = relu(Z)
        activation_cache = A
    if(activation_func == 'sigmoid'):
        A = sigmoid(Z)
        activation_cache = A
        
    cashes = (linear_cache,activation_cache)
    return A,caches
   
def Forward_Propagation(X,parameters,activation_func1,activation_func2,regularization,keep_prob=0.7):
   '''Feed Forward and store the variabes in cache that are required at back propagation'''
   '''Required:
   compute_Activation Function:Function that computes the activation w.r.t relu or
   sigmoid.
   activation_func1 = Activation function used in the hidden layers 
   activation_func2 = Activation function used at the output layers'''
   '''parameters = dictionary of parameters i.e.,Weights and Biases at each layer '''
   A_prev = X
   caches = []
   L = len(parameters)//2
   if (regularization == 'dropout'):
       for l in range (1,L):
            A_prev = A
            D1 = np.random.rand(A_prev.shape[0],A_prev.shape[1])
            D1 = (D1<keep_prob).astype(int)
            A = np.multiply(A_prev,D1)
            A /= keep_prob
            A,cache = compute_Activation(parameters['W'+str(l)],A_prev,parameters['b'+str(l)],activation_func1)   
            caches.append(cache)
   else:
       for l in range (1,L):
              A_prev = A
              A,cache = compute_Activation(parameters['W'+str(l)],A_prev,parameters['b'+str(l)],activation_func1)   
              caches.append(cache)
    
   AL,cache = compute_Activation(parameters['W'+str(l)],A_prev,parameters['b'+str(l)],activation_func2)   
   caches.append(cache)
   return AL,caches


def computeCost(AL,y,parameters,regularization_type,_lambda=0):
   '''Compute the cost
   Required:
   AL : Activation values of the last layer
   Parameters= Dictionary of parameters i.e., weights and biases
   regularization_type:sigmoid or relu
   _lambda:value of lambda for l2 regularization'''
   m = y.shape[1]
   if (regularization_type == 'l2'):
      cost_without_regularization = -(1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)),keepdims=True)
      regularized_Sum = 0
      for l in range (1,(len(parameters)//2)+1):
          print(np.sum(np.sum(np.square(parameters['W'+str(l)]),axis=1)))
          regularized_Sum += np.sum(np.sum(np.square(parameters['W'+str(l)]),axis=1))
      cost = cost_without_regularization + (1/m)*(_lambda/2)*regularized_Sum
   if (regularization_type == 'dropout'):
         cost = -(1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)),keepdims=True)
   
   return cost
#Test Case
A3 = np.array([[ 0.40682402,0.01629284,0.16722898,0.10118111,0.40682402]])
Y = np.array([[1,1,0,1,0]])
parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
       [-1.07296862,  0.86540763, -2.3015387 ]]), 'b1': np.array([[ 1.74481176],
       [-0.7612069 ]]), 'W2': np.array([[ 0.3190391 , -0.24937038],
       [ 1.46210794, -2.06014071],
       [-0.3224172 , -0.38405435]]), 'b2': np.array([[ 1.13376944],
       [-1.09989127],
       [-0.17242821]]), 'W3': np.array([[-0.87785842,  0.04221375,  0.58281521]]), 'b3': np.array([[-1.10061918]])}
cost = computeCost(A3,Y,parameters,'l2',0.1)

def compute_dZ(dA,Activation_Cache,activation):
    '''Compute dZ usinf=g chain rule i.e.,dZ*dA. Reason for this function was
    to compute the derivative of activation functions'''
    '''Required:
       dA : Derivative of cost w.r.t A
       Activation_Cache:Values of the perceptrons after applying the activation functions
    activation= sigmoid or relu'''
    if (activation == 'sigmoid'):
        dZ = np.multiply(np.multiply(Activation_Cache,(1-Activation_Cache)),dA)
    
    if(activation=='relu'):
        dZ=np.multiply(dA,np.int64(Activation_Cache>0))  
    return dZ
#Test Case
Y = np.array([[1,1,0,1,0]])
AL = np.array([[0.9,0.9,0.9,0.9,0.4]])
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
dZ=compute_dZ(dAL,AL,'relu') 

def compute_derivatives(dZ,cache,_lambda):
    ''' To Compute dW,dB and dA that is required for the previous layer
    Required:
    cache:-
    dZ:derivative of the cost w.r.t z
    A_prev:X or the perceptrons of previous layer
    W: Weights of the current layer'''
    W,A_prev,b = cache
    m = A_prev.shape[1]
    dW = (1/m)*(np.dot(dZ,A_prev.T)) + (_lambda/m)*np.sum(W)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    return dW,db,dA_prev

def linear_activation_backward(dA,cache,activation,_lambda=0.01):
    '''Bridge between computing gradients and updating them
    Required:
        compute_derivatives() function that returns dW,db and dA_prev 
        dA : Derivative of current layer perceptrons
        cache: linear-containg W,A_prev,b in oreder and activation-containing A
        activation: Type of activation function required to compute dZ'''
    Linear_Cache,Activation_Cache = cache
    dZ = compute_dZ(dA,Activation_Cache,activation)
    dW,db,dA_prev = compute_derivatives(dZ,Linear_Cache,_lambda)
    
    return dW,db,dA_prev


def backward_propagation(AL,Y,caches):
    '''Combine all steps of backproppagation for one epoch
    Required:
        linear_activation_backward() function
        AL : Activation values of the last layer
        Y  : expected output
        caches: list off all caches computed at the forward prop time  '''
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    current_cache = caches[L-1]
    dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dW"+str(L)],grads["db"+str(L)],grads["dA"+str(L-1)] = linear_activation_backward(dA,current_cache,'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dW"+str(l)],grads["db"+str(l)],grads["dA"+str(l-1)] = linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')
        
    return grads
    
def initialize_adam(parameters):
    '''Initialize v and w used in adam
    Required:
        parameters: Dictionary of Parameters'''
    L = len(parameters)//2
    v = {}
    s = {}
    for l in range(1,L+1):
        v['dW'+str(l)] = np.zeros([parameters['W'+str(l)].shape[0],parameters['W'+str(l)].shape[1]])
        v['db'+str(l)] = np.zeros([parameters['b'+str(l)].shape[0],parameters['b'+str(l)].shape[1]])
        s['dW'+str(l)] = np.zeros([parameters['W'+str(l)].shape[0],parameters['W'+str(l)].shape[1]])
        s['db'+str(l)] = np.zeros([parameters['b'+str(l)].shape[0],parameters['b'+str(l)].shape[1]])
    return v,s

def update_parameters(parameters,grads,v,s,t,learning_rate,beta1=0.9,beta2=0.99,epsilon=10e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}
    for l in range(1,L+1):
        v['dW'+str(l)] = v['dW'+str(l)]*beta1 + (1-beta1)*grads['dW'+str(l)]
        v['db'+str(l)] = v['db'+str(l)]*beta1 + (1-beta1)*grads['db'+str(l)]
        #Corrected Bias
        v_corrected["dW" + str(l+1)] =  v["dW" + str(l+1)]/(1-beta1**t)
        v_corrected["db" + str(l+1)] =  v["db" + str(l+1)]/(1-beta1**t)
        
        v['dW'+str(l)] = v['dW'+str(l)]*beta2 + (1-beta2)*(grads['dW'+str(l)]**2)
        v['db'+str(l)] = v['db'+str(l)]*beta2 + (1-beta2)*(grads['db'+str(l)]**2)
        #Corrected Bias
        s_corrected["dW" + str(l+1)] =  s["dW" + str(l+1)]/(1-beta1**t)
        s_corrected["db" + str(l+1)] =  s["db" + str(l+1)]/(1-beta1**t)
        
        #Update Parameters
        parameters['dW'+str(l)] = parameters['dW'+str(l)] - learning_rate*(v_corrected['dW'+str(l)]/np.sqrt(s_corrected["dW" + str(l+1)]+epsilon))
        parameters['db'+str(l)] = parameters['db'+str(l)] - learning_rate*(v_corrected['db'+str(l)]/np.sqrt(s_corrected["db" + str(l+1)]+epsilon))

#That's It. Happy Reading.        