# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:19:57 2018

@author: Abdallah
"""

import numpy as np
from sklearn import linear_model
import pandas as pd
train = pd.read_csv('train_all.csv')
x_train = train.loc[:,'B':'R']
y_train = train.loc[:,'class']
clf = linear_model.SGDClassifier()
clf.fit(x_train, y_train)



test = pd.read_csv('test_all.csv')
x_test = test.loc[:,'B':'R']
y_hat = clf.predict(x_test)
ids = test.loc[:,'id']
y_hat = pd.DataFrame(y_hat)

y_hat=pd.concat([ids, y_hat],axis=1)
y_hat.to_csv('cla_test_y.csv', encoding='utf-8')