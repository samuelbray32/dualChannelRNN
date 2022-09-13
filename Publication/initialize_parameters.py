#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:04:56 2022

@author: sam
"""
#import keras
#from keras import backend as K
import numpy as np

def initialize_parameters(model,THETA,n_neuron,n_peptide,connectivity,sigmoid_measure=False):
    
    from tools.lattices import WS_network
    con_mask = connectivity
    synapse_matrix = np.random.normal(.1,1,(n_neuron,n_neuron))#*2e-1
    synapse_matrix *= con_mask
    synapse_matrix /= np.mean(synapse_matrix) #keep the overall synaptic strength similar between connectivities
    synapse_matrix *= 2e-4
    
    pep_prod_rate = np.array([1,1],dtype=float)
    pep_decay_rate = np.array([1,1],dtype=float)/5
    D = np.array([.3,1])*10
    pep_action = np.array([3,-2],dtype=float)*1*.3
    neuron_decay=np.array([1.25])
    b = np.random.normal(0,5,(n_neuron,))
    

    offset=6
    def weight_ind(weights,target):
        for i,w in enumerate(weights):
            if target in w.name:
                return i
        return False
    W_ = THETA.weights
    w = THETA.get_weights()
    w[weight_ind(W_,'D')] = D
    w[weight_ind(W_,'pep_prod_rate')] = pep_prod_rate
    #w[2+offset] = pep_decay_rate
    w[weight_ind(W_,'pep_action')] = pep_action
    w[weight_ind(W_,'connectivity')] = con_mask
    w[weight_ind(W_,'pep_decay_rate')] = pep_decay_rate
    w[weight_ind(W_,'synapse_matrix')] = synapse_matrix
    w[weight_ind(W_,'neuron_decay')] = neuron_decay
    w[weight_ind(W_,'input_')] = b
    #w[1] = neuron_decay
    #w[2] = pep_decay_rate
    THETA.set_weights(w)
    
    initial_condition = model.layers[5]#TODO CHECK layer
    w = initial_condition.get_weights()
    w[0][:n_neuron] = np.random.normal(0,1e-2,n_neuron)
    w[0][n_neuron:] = np.random.exponential(1e-2,n_neuron*n_peptide)
    initial_condition.set_weights(w)
    
    M2 = model.layers[-2]#TODO CHECK layer
    w = M2.get_weights()
    w_out = np.random.normal(0,.05,(1,n_neuron,1))
    while (w_out*b).sum()<0:
        w_out = np.random.normal(0,.05,(1,n_neuron,1))
    if sigmoid_measure:
        w[1] = np.array([-1])
    else:
        w[1] = np.array([.1])
    M2.set_weights(w)
    
    return
