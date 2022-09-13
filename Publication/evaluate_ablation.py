#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:58:07 2022

@author: sam
"""

import pickle
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype']='none'


ablate_fract = np.linspace(0,.5,5)
#ablate_fract = np.sort(np.append(ablate_fract, np.logspace(-3,-.1,7)))
method = 'square'

def eval_ablate(folder,model,ablate_fract=np.linspace(0,.5,5),method='square',
                n_test=128,n_neuron=2500,n_genes=1, batch_size=8):
    #check if available memory, wait in line if not
    import psutil
    import time
#     while psutil.virtual_memory().free<8e9:
#         print('delay')
#         time.sleep(10)
    
    sq = int(n_neuron**.5)
    synapse_mask=True
    #data to compare to
    #conditions = ['WT','022421pc2','tbh','gad','chat','th','tbh+pc2','gad+pc2','chat+pc2']
    tags = ['','_30s']
    #gene = [1,0,1,1,1,1,0,0,0]
    sh = np.array([5,30])*2
    #syn_id = [0,1,2,3,4,5,2,3,4]
    data_name = 'trainData/LDS_response_rnai_0430.pickle'
    with open(data_name,'rb') as f:
        result = pickle.load(f)

    fig, ax = plt.subplots(nrows=2,sharex=True,sharey=True)
    fig2 = plt.figure()
    tp = result['tau']
    ind_t = np.where((tp>-2)&(tp<=10))[0]  
    t_on = np.argmin(tp**2)
    dat = 'WT'
    tot_response = [[] for _ in tags]
    response = [[] for _ in tags]
    from tqdm import tqdm
    for i,abl in tqdm(enumerate(ablate_fract)):
        c = plt.cm.viridis(i/len(ablate_fract))
        print(i,abl)
        for j,tag in enumerate(tags):
            yp = result[dat+tag][:,ind_t]
            
            u_i = np.zeros((n_test,tp.size,1))
            if sh[j]>=1:
                u_i[:,t_on:t_on+int(sh[j])] = 1
            else:
                u_i[:,t_on:t_on+1] = sh[j]
            u_i = u_i[:,ind_t]
            u_gene = np.ones((n_test,n_genes))#*(1-i)
    #        u_ablate = np.ones((n_test,n_neuron))
    #        u_ablate = np.random.uniform(0,1,u_ablate.shape)>abl
            if 'square' in method:
                u0 = np.ones((sq,sq))
                w = int((n_neuron*abl)**.5)
                if w==0:
                    h=0
                else:
                    h= int((n_neuron*abl)/w)
                u0[:w,:h] = 0
                u_ablate = []
                for _ in range(n_test):
                    roll = np.random.randint(0,sq,2)
                    u_ii = u0.copy()
                    u_ii = np.roll(np.roll(u_ii,roll[0],axis=0),roll[1],axis=1)
                    u_ablate.append(u_ii)
                u_ablate = np.reshape(np.array(u_ablate),(n_test,n_neuron))
            else: 
                u_ablate = np.ones((n_test,n_neuron))
                for ii in range(n_test):
                    ind_abl = np.random.choice(np.arange(0,n_neuron),int(np.round(abl*n_neuron)))
                    u_ablate[ii,ind_abl] = 0
            
            try:
                if synapse_mask:
                    u_syn = np.ones_like(u_ablate)
                    y_sim = model.predict([u_i,u_gene,u_ablate,u_syn],batch_size=batch_size)
                else:
                    y_sim = model.predict([u_i,u_gene,u_ablate],batch_size=batch_size)
            except:
                y_sim = model.predict([u_i,u_gene,u_ablate],batch_size=batch_size)
            
            loc = np.where(tp[ind_t]==0)[0][0]
            tot_response[j].append(y_sim[:,loc:loc+10*120].mean(axis=1))
            response[j].append(y_sim.copy())
    #        y_sim = latent_model.predict([u_i,u_gene])
            fig2.gca().scatter([abl],[np.mean(tot_response[j][-1])],c=['red','blue'][j])
            fig2.savefig(f'{folder}ablation_{method}_totalR_temp.png')
        
            if i==0:
                ax[j].plot(tp[ind_t],np.median(yp,axis=0),c='grey',label='data')
                ax[j].fill_between(tp[ind_t],np.percentile(yp,25,axis=0),
                np.percentile(yp,75,axis=0), facecolor='grey',alpha=.2)
                c='r'
            ax[j].plot(tp[ind_t],np.median(y_sim[:,:],axis=0),c=c,label=f'ablate =  {np.round(abl,2)}',lw=1)
            ax[j].fill_between(tp[ind_t],np.percentile(y_sim[:,:],25,axis=0),
              np.percentile(y_sim[:,:],75,axis=0),alpha=.2,facecolor=c)
            ax[j].plot(tp[ind_t],u_i[0])
            ax[0].set_title(dat)
    ax[0].legend()
    fig.suptitle(method)    
    fig2, ax2 = plt.subplots()
    for i,rr in enumerate(tot_response):  
        print(tags[i])
        lo =  np.percentile(np.array(rr),25,axis=1)  
        hi =  np.percentile(np.array(rr),75,axis=1)  
        ax2.plot(ablate_fract,np.mean(np.array(rr),axis=1),label=f'{sh[i]/2}s')
        ax2.fill_between(ablate_fract,lo,hi,alpha=.1)
        ax2.legend()
    plt.ylabel('response')
    plt.xlabel('ablation')
    
    response = np.array(response)
    np.save(f'{folder}ablation_{method}_data.npy',response)
    np.save(f'{folder}ablation_{method}_alpha.npy',ablate_fract)
    fig.savefig(f'{folder}ablation_{method}_trace.svg')
    fig.savefig(f'{folder}ablation_{method}_trace.png')
    fig2.savefig(f'{folder}ablation_{method}_totalR.svg')
    fig2.savefig(f'{folder}ablation_{method}_totalR.png')
    plt.close('all')
    del response
    del tot_response
    del result
    del fig
    del fig2
    return
    
    