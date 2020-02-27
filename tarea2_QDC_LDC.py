# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:51:47 2020

@author: elohe
"""

import scipy.io  as sc
import numpy as np
from numpy import linalg as LA

#Variables
lib = sc.loadmat('datos_wdbc.mat')
data = lib["trn"]["xc"][0, 0] 
labels = lib["trn"]["y"][0, 0] 
sizerow = lib["trn"]["n"][0, 0]
sizecol = lib["trn"]["l"][0, 0]
percent=0.8

#Matriz de covarianza de pruebas
#sigma = np.array([[3,2,8],[2,3,9],[8,9,12]])

def sanear(datos):
    D, V = LA.eig(datos)
    ##Se sanean los valores negativos, y se reconstruye la matriz.
    D[D < 0] = 0.001
    tras = np.transpose(V)
    op = (D*V)
    ops = np.dot(op,tras)
    return ops

## Porcentaje en training
trn_percent = int(np.floor(sizerow * percent)[0, 0])
label_training = labels[0: trn_percent]
    

#Se separa la matriz por clases - C1 y C2
    ##Labels e Indices de las clases 1 y 2
c1_tr_lbl = label_training[label_training == 1]
c1_tr_indx = np.where(label_training==1)[0]

c2_tr_lbl = label_training[label_training == 2]
c2_tr_indx = np.where(label_training==2)[0]

    ##Probabilidades a priori
prob_c1 = np.size(c1_tr_lbl) / np.size(label_training)
prob_c2 = np.size(c2_tr_lbl) / np.size(label_training)

### Data training
training_set = data[0: trn_percent, :]

    ##Dataset clasificado
data_trn_c1 = training_set[c1_tr_indx, :]
data_trn_c2 = training_set[c2_tr_indx, :]

    ##matrices de covarianza
sigma = np.cov(training_set, rowvar=False)
sigma_c1 = np.cov(data_trn_c1, rowvar=False)
sigma_c2 = np.cov(data_trn_c2, rowvar=False)

    ##Se sanean los valores negativos, y se reconstruye la matriz. %%
ssaneado = sanear(sigma)
ssaneado_c1 = sanear(sigma_c1)
ssaneado_c2 = sanear(sigma_c2)

## TEST 
label_test = labels[label_training[0,0] : len(labels), :]
test_percent = int(sizerow - trn_percent)
test_set = data[int(trn_percent): len(data), :]


