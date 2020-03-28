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
    '''Computes the actiavtion based on the type of activation function'''
    Z = np.dot(W,A_prev)+b
    linear_cache = Z
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
   '''activation_func1 = Activation function used in the hidden layers 
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


def computeCost(AL,y,parameters,regularization_type,_lambda):
   '''Compute the cost
   Parameters= Dictionary of parameters i.e., weights and biases'''
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
	if (activation == 'sigmoid'):
		dZ = np.multiply(np.multiply(Activation_Cache,(1-Activation_Cache)),dA)
		
	return dZ
	
Y = np.array([[1,1,0,1,0]])
AL = np.array([[1,1,1,1,0]])
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
dZ=compute_dZ(dAL,Y,'sigmoid') 

def linear_activation_backward(A_L , W , b,activation):
	
	
	
def backward_propagation(X,AL,Y,parameters,cache,_lambda):
	grads = {}
	L = len(cache)
	m = X.shape[1]
	
	for l in range(1,L+1):
		
    