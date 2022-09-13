#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:16:38 2022

@author: sam
"""
import pickle
import numpy as np


def prep_data(synapse_mask_fun,syn_fractions,ablate_fract=.0,light_sample=(-1,15),
              jitter=int(3*120),max_pulse=30, set1=False, single_only=False):
    UU = []
    UU_gene=[]
    UU_syn=[]
    ZZ = []
    
    """
    Single pulse_data
    """
    conditions = ['WT','022421pc2','tbh','gad','chat','th','tbh+pc2','gad+pc2','chat+pc2']
    if single_only: conditions = conditions[:6]
    if set1: conditions=conditions[:2]
    tags = ['','_30s']
    gene = [1,0,1,1,1,1,0,0,0]
    sh = np.array([5,30])*2
    syn_id = [0,1,2,3,4,5,2,3,4]
    #if syn_id is None:
    #    syn_id = synapse_mask_fun(np.array([0,0,.125,.125,.5,.125])*1)
    
    data_name = 'trainData/LDS_response_rnai_0430.pickle'
    with open(data_name,'rb') as f:
        result = pickle.load(f)
    tp = result['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    for i,dat in enumerate(conditions):
        for j,tag in enumerate(tags):
            #find datasets
            yp = []
            search=dat+tag
            for nm in result.keys():
                if nm[-len(search):]==search:# (dat+tag in nm) and (tag in nm) and (not '+' in nm):
                    if ('+' in nm) and  not('+' in dat): continue
                    yp.append(result[nm])
                    print(nm)
            yp = np.concatenate(yp)
            print(dat+tag,yp.shape[0])
            yp = np.median(yp,axis=0)[None,...]
    #        yp = np.median(result[dat+tag],axis=0)[None,...]
            print(dat+tag,yp.shape[0])
            ind_samp = np.random.choice(np.arange(yp.shape[0]),int(max_pulse/len(tags)))
            for k in ind_samp:
                jit_i = np.random.randint(0,jitter)
                u_i = np.zeros((tp.size))
                if sh[j]>=1:
                    u_i[t_on:int(t_on+sh[j])] = 1
                else:
                    u_i[t_on:t_on+1] = sh[j]
                u_i = u_i[ind_t-jit_i]
                
    #            u_i = np.repeat(u_i[ind_t,None],yp.shape[0],axis=-1).T      
                
                ZZ.append(yp[k,ind_t-jit_i][None,...])
                UU.append(u_i[None,:])
                UU_gene.append(np.ones(1,)*gene[i])
                UU_syn.append(synapse_mask_fun(syn_id[i],syn_fractions))
                print(dat+tag,gene[i])
    
    """
    Single STEP data
    """
    conditions_step = ['WT','pc2']
    tags_step = ['_30m2h16bp','_30m2h64bp']
    sh_step=np.array([30,30,])*120
    intensity_step = np.array([16/256,64/256])
    data_name = 'trainData/LDS_response_LONG.pickle'
    with open(data_name,'rb') as f:
        result_step = pickle.load(f)
    tp = result_step['tau']
    ind_t = np.where((tp>light_sample[0])&(tp<=light_sample[1]))[0]
    t_on = np.argmin(tp**2)
    for i,dat in enumerate(conditions_step):
        for j,tag in enumerate(tags_step):
            #find datasets
            yp = []
            for nm in result_step.keys():
                if (dat in nm) and (tag in nm) and (not '+' in nm):
                    yp.append(result_step[nm])
                    print(nm)
            yp = np.concatenate(yp)
            print(dat+tag,yp.shape[0])
            yp = np.median(yp,axis=0)[None,...]
    
            ind_samp = np.random.choice(np.arange(yp.shape[0]),int(max_pulse/len(tags_step)))
            for k in ind_samp:
                jit_i = np.random.randint(0,jitter)
                u_i = np.zeros((tp.size))
                if sh_step[j]>=1:
                    u_i[t_on:int(t_on+sh_step[j])] = intensity_step[j]
                else:
                    u_i[t_on:t_on+1] = sh_step[j]*intensity_step[j]
                u_i = u_i[ind_t-jit_i]    
                
                ZZ.append(yp[k,ind_t-jit_i][None,...])
                UU.append(u_i[None,:].copy())
                UU_gene.append(np.ones(1,)*gene[i])
                UU_syn.append(synapse_mask_fun(syn_id[i],syn_fractions))
                
    UU = np.concatenate(UU)[...,None]
    ZZ = np.concatenate(ZZ)
    UU_gene = np.concatenate(UU_gene)[...,None]
    UU_syn = np.array(UU_syn)
    UU_ablate = (np.random.uniform(size=(UU.shape[0],UU_syn.shape[1]))>ablate_fract).astype(float)
    
    
    ind = np.arange(UU.shape[0])
    np.random.shuffle(ind)
    
    UU = UU[ind]
    ZZ = ZZ[ind]
    UU_gene = UU_gene[ind]
    UU_ablate = UU_ablate[ind]
    UU_syn = UU_syn[ind]
    return UU,ZZ,UU_gene,UU_ablate,UU_syn