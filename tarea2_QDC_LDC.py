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

#Se separa la matriz por clases - C1 y C2
    ##Labels e Indices de las clases 1 y 2
c1_lbl = labels[labels == 1]
c1_indx = np.where(labels==1)[0]

c2_lbl = labels[labels == 2]
c2_indx = np.where(labels==2)[0]

    ##Probabilidades a priori
prob_c1 = np.size(c1_lbl) / sizerow
prob_c2 = np.size(c2_lbl) / sizerow

    ##Dataset clasificado
data_c1 = data[c1_indx, :]
data_c2 = data[c2_indx, :]

    ##matrices de covarianza
sigma_c1 = np.cov(data_c1)
sigma_c2 = np.cov(data_c2)

    ##Se sanean los valores negativos, y se reconstruye la matriz. %%
ssaneado_c1 = sanear(sigma_c1)
ssaneado_c2 = sanear(sigma_c2)


