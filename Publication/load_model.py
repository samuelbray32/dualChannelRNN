#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:31:16 2022

@author: sam
"""

import keras
from analysisScripts.neuralModel_paperMethod.keras_dynamicLayers import trainableInput
from analysisScripts.dualNet.keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
model = keras.models.load_model(f'{folder}trained_model',
                                  custom_objects={'trainableInput':trainableInput,
                                                  'rnnCell_dualNet_':rnnCell_})