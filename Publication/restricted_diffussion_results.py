# In[]

import numpy as np
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pickle

# In[]

# In[]

import os
folder= "/media/sam/internal_2/behavior/dualNet/v051122_STAGE3_deepScreen/"
folder_restrict= "/media/sam/internal_2/behavior/dualNet/v051122_STAGE3_deepScreen_DIFFUSSION_RESTRICTED_STAGE2/"
print(os.listdir(folder))

from model import trainableInput
from keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()


phenotype = []
D_length = []
k = []
beta = []

phenotype_restrict = []
D_length_restrict = []
error = []
error_restrict = []

#model settings for evaluate
syn_fractions=np.array([0,0,.125,.125,.5,.125])
n_neuron = 2500
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

os.chdir("/media/sam/internal_2/behavior/cloudTraining_091222/")
from evaluate_fit_error import evaluate_fit

model = None
model_restrict = None
from tqdm import tqdm
for f in tqdm(os.listdir(folder)[:]):
    if not os.path.isdir(f'{folder}{f}'):
        continue
    if not os.path.isdir(f'{folder_restrict}{f}'):
        continue
    if model is None:
        model = keras.models.load_model(f'{folder}{f}/trained_model',
                                            custom_objects={'trainableInput':trainableInput,
                                                        'rnnCell_dualNet_':rnnCell_})
        model_restrict = keras.models.load_model(f'{folder_restrict}{f}/model',
                                            custom_objects={'trainableInput':trainableInput,
                                                        'rnnCell_dualNet_':rnnCell_})
    else:
        model.load_weights(f'{folder}{f}/trained_model')
        model_restrict.load_weights(f'{folder_restrict}{f}/model')

    
    
    # # get diffusion rate of model
    # d_i = np.sqrt(model.get_weights()[1]/model.get_weights()[3])
    # d_i_restrict = np.sqrt(model_restrict.get_weights()[1]/model_restrict.get_weights()[3])

    try:
        r_i = np.load(f'{folder}{f}/ablation_squareDeepScreen_data.npy')
        r_i_restrict = np.load(f'{folder_restrict}{f}/ablation_squareDeepScreen_data.npy')
    except:
        continue

    #get error for restrict model. calculate if not present
    import pandas as pd
    if not os.path.exists(f'{folder_restrict}{f}/fit_error.pkl'):
        #run the error-evaluating fit if not present
        evaluate_fit(model_restrict,synapse_mask_fun,syn_fractions,
                     n_genes=model.layers[-4].cell.n_genes,
                     n_neuron=model.layers[-4].cell.n_neuron,
                     folder=f'{folder_restrict}{f}/')
    if True:#not os.path.exists(f'{folder}{f}/fit_error.pkl'):
        #run the error-evaluating fit if not present
        evaluate_fit(model,synapse_mask_fun,syn_fractions,
                     n_genes=model.layers[-4].cell.n_genes,
                     n_neuron=model.layers[-4].cell.n_neuron,
                     folder=f'{folder}{f}/')
    error_restrict.append(pd.read_pickle(f'{folder_restrict}{f}/fit_error.pkl'))
    error.append(pd.read_pickle(f'{folder}{f}/fit_error.pkl'))

    alpha = np.load(f'{folder}{f}/ablation_squareDeepScreen_alpha.npy')
    alpha_restrict = np.load(f'{folder_restrict}{f}/ablation_squareDeepScreen_alpha.npy')

    integrate = 10
    sample = np.arange(240,240+integrate*120)

    r_tot = np.mean(r_i[:,:,:,sample].mean(axis=-1),axis=-1)
    delta_r_tot = r_tot[:,:] - r_tot[:,0][:,None]
    # D_length.append(d_i)
    phenotype.append(delta_r_tot)

    r_tot = np.mean(r_i_restrict[:,:,:,sample].mean(axis=-1),axis=-1)
    delta_r_tot = r_tot[:,:] - r_tot[:,0][:,None]
    # D_length_restrict.append(d_i_restrict)
    phenotype_restrict.append(delta_r_tot)

    beta.append(float(f.split('beta')[1].split('_')[0]))
    k.append(float(f.split('k')[1].split('_')[0]))
    # break
    # del model

# In[]
fig = plt.figure()
e_avg = []
e_avg_restrict = []
for e in error_restrict:
    e_avg_restrict.append(np.mean(e['MSE']))
for e in error:
    e_avg.append(np.mean(e['MSE']))
plt.violinplot([e_avg,e_avg_restrict],showmeans=True)
e_avg = np.array(e_avg)
plt.xticks([1,2],labels=['original model','restricted_difussion'])
plt.ylabel('MSE')
plt.rcParams['svg.fonttype']='none'
fig.savefig('/home/sam/Desktop/restricted_difussion_model_error.png')
fig.savefig('/home/sam/Desktop/restricted_difussion_model_error.svg')

# In[]
ind = np.where(e_avg[:-1]<.05)[0]
control_plot = np.array(phenotype)[ind,1,:10].max(axis=-1)
restrict_plot = np.array(phenotype_restrict)[ind,1,:10].max(axis=-1)

plt.violinplot([control_plot,restrict_plot])

# In[]
delta_phenotype = restrict_plot-control_plot
plt.violinplot(delta_phenotype)

# In[]
plt.plot(np.array(phenotype)[0,0,:])


# In[]
fig, ax = plt.subplots(nrows=20,figsize=(10,30))
st =50
for i in range(st,st+len(ax)):
    ax[i-st].plot(np.array(phenotype)[i,0,:])
    ax[i-st].plot(np.array(phenotype_restrict)[i,0,:])
    ax[i-st].set_ylabel(np.round(e_avg[i],3))
# %%
ind = np.where(e_avg[:-1]<.01)[0]
xx = np.ravel(np.array(phenotype)[:,1,:])
yy = np.ravel(np.array(phenotype_restrict)[:,1,:])

# ind = np.where(xx<-.2)[0]
# xx = xx[ind]
# yy = yy[ind]
plt.scatter(xx,yy,alpha=.1)
from scipy.stats import linregress

plt.xlim(-3,4)
plt.ylim(-3,4)
plt.plot([-3,4],[-3,-3+7*.9])
linregress(xx,yy)


# %%
plt.violinplot([error, error_restrict])
