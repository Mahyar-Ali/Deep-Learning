# python3 
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:28:44 2020

@author: mahyar
"""

import numpy as np

#Simple RNN

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_cell_forward(xt,a_prev,parameters):
    
    #Retrieve weights and biases from the parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    #Compute a_next and yt_pred
    a_next = np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    
    #Cache all the necessary variables
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next,yt_pred,cache
    

def rnn_forward(x, a0, parameters):
    
    #Get the needed dimensions
    n_x,m,T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # initialize "a" and "y_pred" with zeros
    a = np.zeros([n_a,m,T_x])
    y_pred = np.zeros([n_y,m,T_x])
    
    #To keep track of the computations at each step
    caches =[]
    
    a_next = a0
    
    for t in range(T_x):
        
        xt = x[:,:,t]
        
        a_next,yt_pred,cache = rnn_cell_forward(xt,a_next,parameters)
        
        #Store the activations and output at each time step
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches