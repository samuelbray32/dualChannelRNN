
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from model import reduce_and_lock_diffusion
from keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
from model import trainableInput
from prep_data import prep_data

def fit_script_reduced_diffussion(old_model,folder):
    #Just hard-coding these
    noise = .001
    syn_fractions=np.array([0,0,.125,.125,.5,.125])
    n_neuron = old_model.layers[-4].cell.n_neuron
    n_genes = old_model.layers[-4].cell.n_genes
    # make a new model with restricted peptide diffussion
    print('CREATING NEW MODEL')
    model = reduce_and_lock_diffusion(old_model)
    model.save(f'{folder}model')
    K.clear_session()
    model = keras.models.load_model(f'{folder}model',
                            custom_objects={'trainableInput':trainableInput,
                                          'rnnCell_dualNet_':rnnCell_})       
    opt = keras.optimizers.Adam(learning_rate=5e-4,clipnorm=1e-3)
    model.compile(optimizer=opt,loss='mse')
    latent_model = keras.models.Model(inputs=model.inputs,outputs=[model.layers[-3].input])
    print("TRAINING NEW MODEL")
    ablate_fract = .0
    light_sample=(-1,25)
    jitter = int(3*120) #noise to add to the initial timepoint to prevent overfitting of model
    max_pulse = 25
    def lr_scheduler(epoch,lr):
        return 5e-4 #3e-5+1e-4*np.exp(-epoch/3)
    calls = [keras.callbacks.LearningRateScheduler(lr_scheduler)]
    #train on everything
    def synapse_mask_fun(ID,fractions,):
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
    
    UU,ZZ,UU_gene,UU_ablate,UU_syn = prep_data(synapse_mask_fun,syn_fractions,
                                               ablate_fract,light_sample,
                                               jitter,max_pulse,set1=False)
    model.fit(x=[UU,UU_gene,UU_ablate,UU_syn],y=ZZ,epochs=15,shuffle=True,
              callbacks=calls,batch_size=8,) ###TODO: change epochs
    model.save(f'{folder}model')
    latent_model.save(f'{folder}latent_model')
    
    ####################################################
    ####################################################
    ############      EVALUATE              ############
    ####################################################
    ####################################################
    print('EVALUATING FIT')
    '''eval fit result'''
    from evaluate_fit import evaluate_fit
    evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes,n_neuron,folder)
    plt.close('all')
    
    
    '''eval ablation effect'''
    from evaluate_ablation import eval_ablate
    n_test= 8#64 #TODO: Up number
    alpha=np.sort(np.append(np.linspace(.01,.3,10),np.linspace(0,.99,10)))
    print('EVALUATING ABLATE SQUARE')
    eval_ablate(folder,model,ablate_fract=alpha,method='square',
                    n_test=n_test,n_neuron=n_neuron,n_genes=n_genes, batch_size=16)
    print('EVALUATING ABLATE RANDOM')
    eval_ablate(folder,model,ablate_fract=alpha,method='none',
                    n_test=n_test,n_neuron=n_neuron,n_genes=n_genes, batch_size=16)
    K.clear_session()
    return