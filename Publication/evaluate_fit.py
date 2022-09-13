#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:36:50 2022

@author: sam
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'



def evaluate_fit(model,synapse_mask_fun,syn_fractions,
                 n_genes,n_neuron,folder):
    #check if available memory, wait in line if not
    import psutil
    import time
    while psutil.virtual_memory().free<8e9:
        print('delay')
        time.sleep(10)
        
    
    #define data to test
    conditions = ['WT','022421pc2','tbh','gad','chat','th','tbh+pc2','gad+pc2','chat+pc2']
    tags = ['','_30s']
    gene = [1,0,1,1,1,1,0,0,0]
    sh = np.array([5,30])*2
    syn_id = [0,1,2,3,4,5,2,3,4]
    data_name = 'trainData/LDS_response_rnai_0430.pickle'
    with open(data_name,'rb') as f:
        result = pickle.load(f)
        
    
#     print('data loaded')
    """SINGLE PULSE FIT RESULT"""
    fig, ax = plt.subplots(nrows=len(tags),ncols=len(conditions),sharex=True,sharey=True)
    if len(conditions)==1:
        ax = ax[:,None]
    tp = result['tau']
    ind_t = np.where((tp>-3)&(tp<=7))[0]  
    t_on = np.argmin(tp**2)
    n_test = 4
    for i,dat in enumerate(conditions):
        for j,tag in enumerate(tags):
            yp = []
            for nm in result.keys():
                check = dat+tag
                if nm[-len(check):]==check:
                    if not(('+' in nm) and (not '+' in check)):
                        print(nm)
                        yp.append(result[nm])
            yp = np.concatenate(yp)[:,ind_t]
            u_i = np.zeros((n_test,tp.size,1))
            if sh[j]>=1:
                print(sh[j])
                u_i[:,t_on:t_on+int(sh[j])] = 1
            else:
                u_i[:,t_on:t_on+1] = sh[j]
            u_i = u_i[:,ind_t]
            u_gene = np.ones((n_test,n_genes))*gene[i]#(1-i)
            u_ablate = np.ones((n_test,n_neuron))
            u_syn = np.array([synapse_mask_fun(syn_id[i],syn_fractions) for _ in range(n_test)])
#             print('begin pred')
            y_sim = model.predict([u_i,u_gene,u_ablate,u_syn],batch_size=4)
#             print('end pred')
            ax[j,i].plot(tp[ind_t],np.median(yp,axis=0),c='grey',label=dat+tag)
            ax[j,i].fill_between(tp[ind_t],np.percentile(yp,25,axis=0),
            np.percentile(yp,75,axis=0), facecolor='grey',alpha=.2)
            ax[j,i].plot(tp[ind_t],np.median(y_sim[:,:],axis=0))
            ax[j,i].fill_between(tp[ind_t],np.percentile(y_sim[:,:],25,axis=0),
              np.percentile(y_sim[:,:],75,axis=0),alpha=.2)
            ax[j,i].plot(tp[ind_t],u_i[0])
            ax[0,i].set_title(dat)
            ax[j,i].legend()
    fig.savefig(f'{folder}pulseFit.svg')
    fig.savefig(f'{folder}pulseFit.png')        
    del result
    del fig
    
    #check if available RAM, wait in line if not
    while psutil.virtual_memory().free<8e9:
        print('delay')
        time.sleep(10)
    conditions_step = ['WT','pc2']
    tags_step = ['_30m2h16bp','_30m2h64bp']
    sh_step=np.array([30,30,])*120
    intensity_step = np.array([16/256,64/256])
    data_name = 'trainData/LDS_response_LONG.pickle'
    with open(data_name,'rb') as f:
        result_step = pickle.load(f)
    """STEP FIT RESULT"""
    fig, ax = plt.subplots(nrows=len(tags_step),ncols=len(conditions_step),sharex=True,sharey=True)
    if len(conditions)==1:
        ax = ax[:,None]
    tp = result_step['tau']
    ind_t = np.where((tp>-1)&(tp<=35))[0]  
    t_on = np.argmin(tp**2)
    n_test = 4
    for i,dat in enumerate(conditions_step):
        for j,tag in enumerate(tags_step):
            #find datasets
            yp = []
            for nm in result_step.keys():
                if (dat in nm) and (tag in nm) and (not '+' in nm):
                    yp.append(result_step[nm])
                    print(nm)
            yp = np.concatenate(yp)[:,ind_t]
            print(dat+tag,yp.shape[0])
    
            u_i = np.zeros((n_test,tp.size,1))
            u_i[:,t_on:t_on+int(sh_step[j])] = intensity_step[j]
            u_i = u_i[:,ind_t]
            u_gene = np.ones((n_test,n_genes))*gene[i]
            u_ablate = np.ones((n_test,n_neuron))
            u_syn = np.array([synapse_mask_fun(syn_id[i],syn_fractions) for _ in range(n_test)])
            y_sim = model.predict([u_i,u_gene,u_ablate,u_syn],batch_size = 4)
    
            ax[j,i].plot(tp[ind_t],np.median(yp,axis=0),c='grey',label=dat+tag)
            ax[j,i].fill_between(tp[ind_t],np.percentile(yp,25,axis=0),
            np.percentile(yp,75,axis=0), facecolor='grey',alpha=.2)
            ax[j,i].plot(tp[ind_t],np.median(y_sim[:,:],axis=0))
            ax[j,i].fill_between(tp[ind_t],np.percentile(y_sim[:,:],25,axis=0),
              np.percentile(y_sim[:,:],75,axis=0),alpha=.2)
            ax[j,i].plot(tp[ind_t],u_i[0])
            ax[0,i].set_title(dat)
            ax[j,i].legend()
    fig.savefig(f'{folder}stepFit.svg')
    fig.savefig(f'{folder}stepFit.png')
    del result_step
    del fig
    return

