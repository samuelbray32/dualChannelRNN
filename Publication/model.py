#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:52:05 2022

@author: sam
"""

import keras
from keras import backend as K
from keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
import tensorflow as tf


class zeroConstraint(keras.constraints.Constraint):
  """Constrains weight tensors to sum to 0."""
  def __init__(self,):
      return
  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean 

####################################
###### CUSTOM LAYERS ###############    
####################################
class trainableInput(keras.layers.Layer):
    #layer that returns it's weight on call 
    #(pretty much gives you an easily manipulable trainable variable)
    def __init__(self, p_shape=(100,),initializer='zeros',**kwargs):
        self.p_shape = p_shape
        self.initializer = initializer
        super(trainableInput, self).__init__(**kwargs)
    def build(self,input_shape):
        self._P = self.add_weight(name='P', 
                    shape=self.p_shape,
                    initializer=self.initializer,#keras.initializers.glorot_uniform(),
                    trainable=True,)
        super(trainableInput, self).build(input_shape)
    def get_config(self): 
        config = {'p_shape': self.p_shape,
                  'initializer':self.initializer,
#                    'n': self.n,
#                  'G': np.array(K.eval(self.G)),
#                  'init_zeros': self.init_zeros,
#                  'noise': self.noise,
                }
        base_config = super(trainableInput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self,inputs):
#        samples = inputs.shape.as_list()[0]
#        print(inputs.shape.as_list()[0])
        out = [-1,]
        out.extend(list(self.p_shape))
#        P = K.tile(self._P,out)
#        return K.reshape(P,(samples))
        n_tot = 1
        for i in self._P.shape.as_list():    
            n_tot *= i
#        dummy = K.flatten(inputs)
        dummy = inputs
        print(dummy.shape.as_list())
        n_in = dummy.shape.as_list()[1]
        if n_tot>n_in:
            dummy = K.tile(dummy,n_tot//n_in+1)
        print(dummy.shape)
        dummy = dummy[:,:n_tot]
        print(dummy.shape)
        dummy = K.reshape(dummy,out)
        dummy = dummy * 0 + self._P
        print(dummy.shape)
        return dummy

    def compute_output_shape(self, input_shape): 
        out = [None,]
        out.extend(list(self.p_shape))
        return tuple(out)
class initial_condition_layer(keras.layers.Layer):
    
    def __init__(self, n=100, n_genes=3, G=None, init_zeros=False, noise=False,
                 gene_specific=False,**kwargs):
        self.n = n
        #check if got passed as list, turn into np arrray if so:
        if isinstance(G,list):
            G=np.array(G)
        #Genetic Mask matrix
        if G is None:
            G = np.zeros((n,n_genes))
            split = np.linspace(0,n,n_genes+1).astype(int)
            for i in range(n_genes):
                st = split[i]
                en = split[i+1]
                G[st:en,i] = 1
        #If ID is passed in a non-one-hot array
        elif len(G.shape)==1:
            G0 = G.copy()
            G = np.zeros((n,n_genes))
            for i,v in enumerate(G0):
                #if motor neuron
                if v==-1:
                    G[i]=1
                #if neural type
                else:
                    G[i,v] = 1
        self.G = tf.constant(G,dtype='float32')
        #Whether this is a layer that just returns a zero tensor or is trainable layer
        self.init_zeros = init_zeros
        #whether to add noise to Q0
        self.noise = noise
        #Whether each knockdown condition learns its own initialization
        self.gene_specific = gene_specific
        super(initial_condition_layer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Creates all trainable weight in mechanistic model
        self._Q0 = self.add_weight(name='Q0', 
                                    shape=(self.n,),
                                    initializer='zeros',#pos_init,#keras.initializers.glorot_uniform(),
                                    trainable=True,)
        if self.gene_specific:
            self._Qgene = self.add_weight(name='Qgene',
                                          shape=(self.G.shape.as_list()[1],self.n),
                                          initializer='zeros',
                                          trainable=True,
                                          )
        super(initial_condition_layer, self).build(input_shape)
    
    def get_config(self):
        config = {'n': self.n,
                  'G': np.array(K.eval(self.G)),
                  'init_zeros': self.init_zeros,
                  'noise': self.noise,
                  'gene_specific': self.gene_specific,
                }
        base_config = super(initial_condition_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    
    def call(self, inputs):
        gamma = inputs
        #conditional genetic matrix
        G = tf.einsum('ij,kj->kij',self.G, gamma)
        G = K.max(G,axis=-1)
        Q = self._Q0
        if self.gene_specific:
            gamma_inv = 1-gamma
            delta = tf.einsum('kj,ji->kij',gamma_inv, self._Qgene)
            delta = K.sum(delta,axis=-1)
            Q = Q + delta
        Q = tf.math.multiply(Q,G)
        if self.init_zeros:
            Q = 0*Q
        if self.noise:
            Q = Q + K.random_normal(tf.shape(Q),0,.03)
        return  Q#TODO: consider noise sampling
        
    def compute_output_shape(self, input_shape): 
        return (None, self.n)    


###########################
##### build the model #####
###########################
def build_model(log_train = False,n_neuron=50**2,
                n_peptide=2,init='glorot_uniform',trainable = [False for i in range(7)],
                magnitude_fit=True,n_genes=1,ablate=True,constrain_measure=False,
                sigmoid_measure=False,connectivity_mask = True,synapse_mask=True,
                noise = 1e-2,clipnorm=None):
    
    sq=int(n_neuron**.5)    

    #stimulus
    U = keras.layers.Input((None,1,),name='U_in')
    #Model RNN
    THETA_cell = rnnCell_(initializer=init,log_train=log_train, weight_train=trainable,
                          n_neuron=n_neuron,n_peptide=n_peptide,n_genes=n_genes,ablate=ablate,
                          return_firing=True,magnitude_fit=magnitude_fit, 
                          connectivity_mask=connectivity_mask, synapse_mask=synapse_mask,noise=noise)
    THETA = keras.layers.RNN(THETA_cell,return_sequences=True,return_state=True)
    
    #Initial condition generator
    U_dummy = keras.layers.Lambda(lambda x: x[:,0,:])(U)
    if log_train:
        initial_condition = trainableInput(p_shape=(THETA_cell.state_size-n_genes-2*n_neuron,),initializer='zeros')
        Q0 = initial_condition(U_dummy)
        Q0 = keras.layers.Lambda(lambda x: tf.exp(x))(Q0)
    else:
        initial_condition = trainableInput(p_shape=(THETA_cell.state_size-n_genes-2*n_neuron,),initializer=init)
        Q0 = initial_condition(U_dummy)
    #    Q0 = keras.layers.Lambda(lambda x: tf.abs(x))(Q0)
    
    #genetic input
    U_gene = keras.layers.Input((n_genes,),name='U_in_gene')
    U_ablate = keras.layers.Input((n_neuron,),name='U_ablate')
    U_syn = keras.layers.Input((n_neuron,),name='U_syn')
    Q0 = keras.layers.Concatenate(axis=1)([U_ablate,U_syn,U_gene,Q0])
    
    #run equilibrium and real model
    U_eq = keras.layers.Lambda(lambda x: 0*x[:,:1200])(U)
    Q0 = THETA(U_eq,initial_state=[Q0,],)[1]
    Q_hat,Q_final = THETA(U,initial_state=[Q0,])
    
    #measurement layers
    M = keras.layers.Lambda(lambda x: K.relu(x[:,:,:n_neuron]))
    act = 'linear'
    if sigmoid_measure:
        act = 'sigmoid'
    constrain = None
    if constrain_measure:
        constrain = zeroConstraint()
    M2 = keras.layers.Conv1D(filters=1,kernel_size=1,input_shape=(None,n_neuron,),activation=act,
                             kernel_initializer='glorot_uniform',data_format='channels_last',
                             kernel_constraint=constrain,kernel_regularizer='l2')
    Z_hat = M(Q_hat)
    Z_hat = M2(Z_hat)
    Z_hat = keras.layers.Lambda(lambda x: K.squeeze(x,-1))(Z_hat)
    
    #In[]
    #full model
    def my_loss(q0,qf):
        def my_loss_(y_true,y_pred):
            eq_loss = 0*keras.losses.mean_squared_error(q0,qf)
            print('eq_loss',eq_loss.shape)
            return keras.losses.mean_squared_error(y_true,y_pred) + eq_loss
    #        return keras.losses.mean_absolute_error(y_true,y_pred) + eq_loss
            #    return keras.losses.mean_absolute_percentage_error(y_true,y_pred)
        return my_loss_
    
    model = keras.models.Model(inputs=[U,U_gene,U_ablate,U_syn],outputs=[Z_hat])
    opt = keras.optimizers.Adam(learning_rate=1e-3,clipnorm=clipnorm)#keras.optimizers.Adadelta(learning_rate=1e-3)#
    model.compile(optimizer=opt,loss='mse',)#my_loss(Q0,Q_final),)
    #model that leaves it in latent space
    latent_model = keras.models.Model(inputs=[U,U_gene,U_ablate,U_syn], outputs=[Q_hat])
    return model, latent_model, THETA
    
########################################################################################################################
# MODEL AUGMENTATION
def map_weights(old, new):
    names=['pep_decay_rate','synapse_matrix','neuron_decay','input_layer',
           'synapse_scale','peptide_scale','D','pep_product_rate','pep_action',
           'connectivity',]
    weight_map = []
    for nm in names:
        i_new = None
        for i,w in enumerate(new):
            if nm in w.name:
                i_new=i
                break
        if i_new is None: continue
        i_old = None
        for i,w in enumerate(old):
            if nm in w.name: 
                i_old=i
                break
        if i_old is None:
            print (f'ERROR: {nm} not found in original weights')
            continue
        weight_map.append((i_old, i_new))
    return weight_map

def augment_model_trainability(model):
    #make new fully trainable RNN cell
    THETA_old = model.layers[-4]
    train_new = [True for _ in THETA_old.cell.weight_train]
    THETA_new_cell = rnnCell_(weight_train=train_new, initializer=THETA_old.cell.init,
                              log_train=THETA_old.cell.log_train, n_neuron=THETA_old.cell.n_neuron,
                             n_peptide=THETA_old.cell.n_peptide, n_genes=THETA_old.cell.n_genes,
                             ablate=THETA_old.cell.ablate, return_firing=THETA_old.cell.return_firing,
                             magnitude_fit=THETA_old.cell.magnitude_fit,
                             connectivity_mask=THETA_old.cell.connectivity_mask, 
                             synapse_mask=THETA_old.cell.synapse_mask, noise=THETA_old.cell.noise)
    print(THETA_new_cell.weight_train, train_new)
    THETA_new = keras.layers.RNN(THETA_new_cell,return_sequences=True,return_state=True)
    
    #make new model
    U_in, U_in_gene, U_ablate, U_syn = model.inputs
    Q0 = model.layers[7].output
    U_eq = model.layers[6].output
    Q0 = THETA_new(U_eq,initial_state=[Q0,],)[1]
    Q_hat,Q_final = THETA_new(U_in,initial_state=[Q0,])    
    Z = model.layers[-3](Q_hat)
    Z = model.layers[-2](Z)
    Z = model.layers[-1](Z)
    def my_loss(q0,qf):
        def my_loss_(y_true,y_pred):
            eq_loss = 0*keras.losses.mean_squared_error(q0,qf)
            print('eq_loss',eq_loss.shape)
            return keras.losses.mean_squared_error(y_true,y_pred) + eq_loss
    #        return keras.losses.mean_absolute_error(y_true,y_pred) + eq_loss
            #    return keras.losses.mean_absolute_percentage_error(y_true,y_pred)
        return my_loss_ 
    model_new = keras.models.Model(inputs=model.inputs, outputs=[Z])
    opt = keras.optimizers.Adam(learning_rate=1e-3)#keras.optimizers.Adadelta(learning_rate=1e-3)#
    model_new.compile(optimizer=opt,loss='mse')
    
    #set weights of new RNN
    w_old = THETA_old.get_weights()
    w_map = map_weights(THETA_old.weights,THETA_new.weights)
    w_new = [None for _ in w_map]
    for m in w_map:
        w_new[m[1]] = w_old[m[0]]
    model_new.layers[-4].set_weights(w_new)
    return model_new

