# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:51:47 2020

@author: elohe
"""

import scipy.io  as sc
import numpy as np
from numpy import linalg as LA
import math

#Variables
lib = sc.loadmat('datos_wdbc.mat')
data = lib["trn"]["xc"][0, 0] 
labels = lib["trn"]["y"][0, 0] 
sizerow = lib["trn"]["n"][0, 0]
sizecol = lib["trn"]["l"][0, 0]


nombre = int(input("Seleccione el porcentaje: (1)70-30, (2)80-20, (3)90-10: "))
if nombre ==  1:
    percent = 0.7
elif nombre == 2:
    percent = 0.8
else:
    percent = 0.9

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

def mediaDeMatriz(matriz):
    media = []
    for i in range(len(data_trn_c1[1])):
        acumulado = np.mean(data_trn_c1[:, i])
        media.append(acumulado)
    
    return media

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

## Se calcula QDC y LDC
QDC_c1 = []
QDC_c2 = []
LDC_c1 = []
LDC_c2 = []


testlen = len(test_set[:, 0])

m_c1 = mediaDeMatriz(data_trn_c1)
m_c2 = mediaDeMatriz(data_trn_c2)

E_inv_TOT = LA.inv(ssaneado) 
E_inv_C1 = LA.inv(ssaneado_c1)
E_inv_C2 = LA.inv(ssaneado_c2)

E_C1 = ssaneado_c1
E_C2 = ssaneado_c2

#Calculo del sumando 1 de QDC
X = test_set[0, :]

def sumandosQDC(X, inversa, median, saneada, prob_clase):
    X_trans = np.transpose(X)
    median_trans = np.transpose(median)
    a = np.dot(X, inversa)
    suma1 = -0.5 * np.dot(a, X_trans)
    
    b = np.dot(median, inversa) 
    suma2 =  np.dot(b, X_trans)

    suma3 = - 0.5 * np.dot(b, median_trans)
    
    suma4 = - 0.5 * math.log(np.linalg.det(saneada))
    suma5= math.log(prob_clase)
    suma_total = suma1 + suma2 + suma3 + suma4 + suma5
    
    return suma_total

def sumandosLDC(X, median, inversa, prob_clase):
    trans_X2 = np.transpose(X)
    trans_median = np.transpose(median)
    
    d = np.dot(median, inversa)
    suma2_1 = np.dot(d, trans_X2)
    
    suma2_2 = -0.5 * np.dot(d, trans_median)
    
    suma2_3 = math.log(prob_clase)
    suma_total_2 = suma2_1 + suma2_2 + suma2_3
    
    return suma_total_2


for i in range(testlen):
    X = test_set[i, :]
    QDC_suma_c1 = sumandosQDC(X, E_inv_C1, m_c1, E_C1, prob_c1)
    QDC_suma_c2 = sumandosQDC(X, E_inv_C2, m_c2, E_C2, prob_c2)
    QDC_c1.append(QDC_suma_c1)
    QDC_c2.append(QDC_suma_c2)

for i in range(testlen):
    X_2 = test_set[i, :]
    LDC_suma_c1 = sumandosLDC(X_2, m_c1, E_inv_TOT, prob_c1)
    LDC_suma_c2 = sumandosLDC(X_2, m_c2, E_inv_TOT, prob_c2)
    LDC_c1.append(LDC_suma_c1)
    LDC_c2.append(LDC_suma_c2)

