#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:54:14 2022

@author: sam
"""

import numpy as np
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
def check(f, requirements):
    for r in requirements:
        if r in f: 
            continue
        else: 
            return False
    return True
def get_beta(f):
   start = f.find('beta')
   end = f[start:].find('_')
   return float(f[start+4:end+start])

def get_k(f):
   start = f.find('k')
   end = f[start:].find('_')
   return float(f[start+1:end+start])
   

# In[]
'''
beta sweep
'''
integrate=5
folder = f'./analysisScripts/dualNet/bigRun/STAGE_1/'
noise = 'noise0.001'
k=32
requirements = [noise,f'k{k}']
difference = True

beta = []
response = []

t = np.arange(-2,10,1/120)
sample = np.arange(240,240+integrate*120)
for f in os.listdir(folder):
    if not os.path.isdir(folder+f+'/'): continue
    if not check(f, requirements): continue 
    try:
        r_i = np.load(f'{folder}{f}/ablation_square_data.npy')
    except:
        continue
    beta_i = get_beta(f)    
    beta.append(beta_i)
    print(r_i.shape)
    r_i = np.mean(r_i[:,:,:,sample].mean(axis=-1),axis=-1)
    if difference:
        r_i=r_i-r_i[:,0][...,None]
    response.append(r_i)
    alpha = np.load(f'{folder}{f}/ablation_square_alpha.npy')
    
beta = np.array(beta)
order = np.argsort(-beta)
beta = beta[order]
response = np.array(response)[order]



    

#response = np.mean(response[:,:,:,:,sample].mean(axis=-1),axis=-1)
#if difference:
#    response=response-response[:,:,0][...,None]
fig, ax = plt.subplots(nrows=2,sharex=True,sharey=True)
for i, a in enumerate(ax):
    if difference:
        rng = np.abs(response).max()/3
        a.imshow(response[:,i,:],cmap='RdBu',clim=(-rng,rng)) 
    else:
        a.imshow(response[:,i,:]) 
    
    a.set_yticks(np.arange(beta.size))
    a.set_yticklabels(np.round(np.log10(beta),2))    
ax[1].set_xticks(np.arange(alpha.size))
ax[1].set_xticklabels(np.round(alpha,2),rotation=90)

ax[1].set_xlabel('ablation')
ax[1].set_ylabel('log10 beta')
plt.suptitle(f'k = {k}')

# In[]
'''
k sweep
'''
integrate=10
folder = f'./analysisScripts/dualNet/bigRun/'
noise = 'noise0.001'
beta='0'
requirements = [noise,f'beta{beta}']
difference = True

k = []
response = []
t = np.arange(-2,10,1/120)
sample = np.arange(240,240+integrate*120)
for f in os.listdir(folder):
    if not check(f, requirements): continue 
    k_i = get_k(f)    
    k.append(k_i)
    r_i = np.load(f'{folder}{f}/ablation_square_data.npy')
    r_i = np.load(f'{folder}{f}/ablation_square_data.npy')
    r_i = np.mean(r_i[:,:,:,sample].mean(axis=-1),axis=-1)
    if difference:
        r_i=r_i-r_i[:,0][...,None]
    response.append(r_i)
    alpha = np.load(f'{folder}{f}/ablation_square_alpha.npy')
#    print(response[-1].shape)
k = np.array(k)
order = np.argsort(-k)
k = k[order]
response = np.array(response)[order]
    

fig, ax = plt.subplots(nrows=2,sharex=True,sharey=True)
for i, a in enumerate(ax):
    if difference:
        rng = np.abs(response).max()/3
        a.imshow(response[:,i,:],cmap='RdBu',clim=(-rng,rng)) 
    else:
        a.imshow(response[:,i,:]) 
    
    a.set_yticks(np.arange(k.size))
    a.set_yticklabels(k)    
ax[1].set_xticks(np.arange(alpha.size))
ax[1].set_xticklabels(np.round(alpha,2),rotation=90)

ax[1].set_xlabel('ablation')
ax[1].set_ylabel('node degree')
plt.suptitle(f'Beta = {beta}')
 
    
    
    
    
    
    