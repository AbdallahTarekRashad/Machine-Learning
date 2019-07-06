
import pandas as pd
import numpy as np
from sklearn import  linear_model



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


reg = linear_model.LinearRegression()
 
# train the model using the training sets
reg.fit(x, y)
 
# regression coefficients
print('Coefficients: \n', reg.coef_)
 
# variance score: 1 means perfect prediction
 
test =pd.read_csv('reg_test.csv')
inst = test.loc[:,'instant']
test =test.drop(['dteday'], axis=1)
one = np.ones([3476,1])
one = pd.DataFrame(one,dtype=int)
test=pd.concat([one, test],axis=1)
test=np.matrix(test)

y_hat = reg.predict(test)

