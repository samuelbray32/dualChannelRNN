#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:18:43 2022

@author: sam

BRANCHED FROM 04.19.22
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt

def fit_script_stage123(folder,connectivity, noise=1e-4,syn_fractions=np.array([0,0,.05,.05,.33,.05])):
#    syn_fractions /= syn_fractions.sum()
    #folder='./analysisScripts/dualNet/bigRun/TEST/'
    
    if not os.path.isdir(folder):
            os.mkdir(folder)
    np.save(f'{folder}syn_fractions.npy',np.array(syn_fractions))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True#False
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    K.clear_session()
    
    #In[]
    '''Build the model'''
    ####################################################
    log_train = False
    reduce_training_variables=0#True
    n_neuron=50**2
    n_peptide=2
    sq=int(n_neuron**.5)
    init='glorot_uniform'
    trainable = [False for i in range(7)]
    trainable[6] = True
    trainable[2] = True
    trainable[4] = True
    trainable[5] = True
    magnitude_fit=True
    
    n_genes=1
    ablate=True
    constrain_measure=False
    sigmoid_measure=False
    connectivity_mask = True
    synapse_mask=True
    
    ####################################################
    from model import build_model
    model, latent_model, THETA = build_model(log_train,n_neuron,n_peptide,init,trainable,
                    magnitude_fit,n_genes,ablate,constrain_measure,
                    sigmoid_measure,connectivity_mask,synapse_mask,
                    noise)
    
    '''set parameters'''
    from initialize_parameters import initialize_parameters
    initialize_parameters(model,THETA,n_neuron,n_peptide,connectivity,sigmoid_measure)
    
    
    
    """################################################
    ####################################################
    ####################################################
    STAGE 1
    ####################################################
    ####################################################
    ####################################################"""
    '''prep data'''
    ####################################################
    ablate_fract = .0
    light_sample=(-1,25)
    jitter = int(3*120) #noise to add to the initial timepoint to prevent overfitting of model
    max_pulse = 20
    ####################################################
    def synapse_mask_fun(ID,fractions):
        if not type(ID) is list:
            ID = [ID]
        mask = []
        ind = np.arange(n_neuron)
        np.random.seed(0)
        np.random.shuffle(ind)
        loc=0
        for f in fractions:
            n = int(f*n_neuron)
            x = np.ones(n_neuron)
            x[ind[loc:loc+n]] = 0
            mask.append(x.copy())
            loc += n
        return (np.array(mask)[ID]).min(axis=0)
    
    """fit"""
    calls=[]
    def lr_scheduler(epoch,lr):
        return 1e-4+1e-3*np.exp(-epoch/15)
    calls = [keras.callbacks.LearningRateScheduler(lr_scheduler)]
    #train on data
    from prep_data import prep_data
    #train on everything (SHORT)
    light_sample=(-1,12)
    UU,ZZ,UU_gene,UU_ablate,UU_syn = prep_data(synapse_mask_fun,syn_fractions,
                                               ablate_fract,light_sample,
                                               jitter,max_pulse,set1=False)
    model.fit(x=[UU,UU_gene,UU_ablate,UU_syn],y=ZZ,epochs=10,shuffle=True,
              callbacks=calls,batch_size=6,)
    '''save model'''
    model.save(f'{folder}trained_model_s1')
    latent_model.save(f'{folder}trained_model_latent_s1')
    '''checkpoint eval'''
    from evaluate_fit import evaluate_fit
    print('start eval')
    evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes,n_neuron,folder)
    plt.close('all')
    #train on everything (LONG)
    light_sample=(-1,25)
    UU,ZZ,UU_gene,UU_ablate,UU_syn = prep_data(synapse_mask_fun,syn_fractions,
                                               ablate_fract,light_sample,
                                               jitter,max_pulse,set1=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=5e-4)
    calls = [keras.callbacks.LearningRateScheduler(lr_scheduler),early_stop]
    model.fit(x=[UU,UU_gene,UU_ablate,UU_syn],y=ZZ,epochs=40,shuffle=True,
              callbacks=calls,batch_size=4,)
    '''save model'''
    model.save(f'{folder}trained_model_s1')
    latent_model.save(f'{folder}trained_model_latent_s1')
    print('start eval')
    evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes,n_neuron,folder)
    plt.close('all')
    
    
    """################################################
    ####################################################
    ####################################################
    STAGE 2
    ####################################################
    ####################################################
    ####################################################"""
    ablate_fract = .0
    light_sample=(-1,25)
    jitter = int(3*120) #noise to add to the initial timepoint to prevent overfitting of model
    max_pulse = 30
    calls=[]
    def lr_scheduler(epoch,lr):
        return 3e-5#1e-5+9e-5*(epoch<20)+2e-4*(epoch<5)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=5e-4)
    calls = [keras.callbacks.LearningRateScheduler(lr_scheduler),early_stop]
    #train on everything
    UU,ZZ,UU_gene,UU_ablate,UU_syn = prep_data(synapse_mask_fun,syn_fractions,
                                               ablate_fract,light_sample,
                                               jitter,max_pulse,set1=False)
    model.fit(x=[UU,UU_gene,UU_ablate,UU_syn],y=ZZ,epochs=10,shuffle=True,
              callbacks=calls,batch_size=4,)
    
    '''save model'''
    model.save(f'{folder}trained_model_s2')
    latent_model.save(f'{folder}trained_model_latent_s2')
    evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes,n_neuron,folder)
    plt.close('all')
    
    """################################################
    ####################################################
    ####################################################
    STAGE 3
    ####################################################
    ####################################################
    ####################################################"""
    #augment trainable parameters
    from model import augment_model_trainability
    from model import trainableInput
    from keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
    model_new = augment_model_trainability(model)
    model_new.save(f'{folder}aug_model')
    K.clear_session()
    model = keras.models.load_model(f'{folder}aug_model',
                            custom_objects={'trainableInput':trainableInput,
                                          'rnnCell_dualNet_':rnnCell_})       
    opt = keras.optimizers.Adam(learning_rate=1e-3,clipnorm=1e-3)
    model.compile(optimizer=opt,loss='mse')
    latent_model = keras.models.Model(inputs=model.inputs,outputs=[model.layers[-3].input])
    
    ablate_fract = .0
    light_sample=(-1,25)
    jitter = int(3*120) #noise to add to the initial timepoint to prevent overfitting of model
    max_pulse = 25
    def lr_scheduler(epoch,lr):
        return 3e-5+1e-4*np.exp(-epoch/3)
    calls = [keras.callbacks.LearningRateScheduler(lr_scheduler)]
    #train on everything
    UU,ZZ,UU_gene,UU_ablate,UU_syn = prep_data(synapse_mask_fun,syn_fractions,
                                               ablate_fract,light_sample,
                                               jitter,max_pulse,set1=False)
    model.fit(x=[UU,UU_gene,UU_ablate,UU_syn],y=ZZ,epochs=10,shuffle=True,
              callbacks=calls,batch_size=4,)
    model.save(f'{folder}trained_model_s3')
    latent_model.save(f'{folder}trained_model_latent_s3')
    
    ####################################################
    ####################################################
    ############      EVALUATE              ############
    ####################################################
    ####################################################
    
    '''eval fit result'''
    from evaluate_fit import evaluate_fit
    evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes,n_neuron,folder)
    plt.close('all')
    
    
    '''eval ablation effect'''
    from evaluate_ablation import eval_ablate
    n_test=64
    alpha=np.sort(np.append(np.linspace(.01,.3,10),np.linspace(0,.99,10)))
    eval_ablate(folder,model,ablate_fract=alpha,method='square',
                    n_test=n_test,n_neuron=n_neuron,n_genes=n_genes, batch_size=16)
    eval_ablate(folder,model,ablate_fract=alpha,method='none',
                    n_test=n_test,n_neuron=n_neuron,n_genes=n_genes, batch_size=16)
    K.clear_session()
    return




