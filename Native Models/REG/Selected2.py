# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:18:42 2018

@author: Abdallah
"""

import pandas as pd
import numpy as np
train = pd.read_csv('reg_train.csv')
train =train.drop(['dteday','casual','registered'], axis=1)
x_train = train.loc[:,'instant':'windspeed']
y_train = train.loc[:,'cnt']
one = np.ones([13903,1])
one = pd.DataFrame(one,dtype=int)
x=pd.concat([one, x_train],axis=1)
x=np.matrix(x)
y = np.matrix(y_train)
y = y.reshape(13903,1)
x_t=x.reshape(14,13903)
xTx = np.dot(x_t,x)
xTx_Inv = np.linalg.inv(xTx)
xTx_Inv_xT = np.dot(xTx_Inv,x_t)
Beta = np.dot(xTx_Inv_xT,y)
print(Beta)
test =pd.read_csv('reg_test.csv')
inst = test.loc[:,'instant']
test =test.drop(['dteday'], axis=1)
one = np.ones([3476,1])
one = pd.DataFrame(one,dtype=int)
test=pd.concat([one, test],axis=1)
test=np.matrix(test)
y_hat = np.dot(test,Beta)
y_hat = pd.DataFrame(y_hat)
y_hat = pd.concat([inst , y_hat] , axis = 1)
y_hat.to_csv('reg_test_y.csv', encoding='utf-8')