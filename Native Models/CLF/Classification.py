import numpy as np 
import pandas as pd


def mycov(x):
    x = x - x.mean(axis=0)
    U, s, V = np.linalg.svd(x, full_matrices = 0)
    C = np.dot(np.dot(V.T,np.diag(s**2)),V)
    return C / (x.shape[0] - 1)

n=145186

tr_data=pd.read_csv("train_all.csv")
test=pd.read_csv("test_all.csv")
ids = test.loc[:,'id']
test=test.drop(['id'], axis=1)


test = np.matrix(test)
tr_data=tr_data.drop(['id'], axis=1)



class1_data=tr_data[tr_data['class']==1]
class1_data=class1_data.drop(['class'], axis=1)
class1_data=np.matrix(class1_data)
mean1=class1_data.mean(0)
s1 = mycov(class1_data)


'''
mean1=np.dot(np.ones([40687,1]),mean1)
class1_c=class1_data-mean1
class1_c_t=class1_c.reshape(3,40687)
sigma1=np.dot(class1_c_t,class1_c)
sigma1=sigma1 * (1/(n-1))
'''

print(s1.shape)



class2_data=tr_data[tr_data['class']==2]
class2_data=class2_data.drop(['class'], axis=1)
class2_data=np.matrix(class2_data)
s2 = mycov(class2_data)
mean2=class2_data.mean(0)
mean1 = mean1.reshape(3,1)
mean2 = mean2.reshape(3,1)

'''
mean2=np.dot(np.ones([104499,1]),mean2)
class2_c=class2_data-mean2
class2_c_t=class2_c.reshape(3,104499)
sigma2=np.dot(class2_c_t,class2_c)
sigma2=sigma2 * (1/(n-1))

'''

print(s2.shape)


sigma1_inv =np.linalg.inv(s1)
sigma2_inv =np.linalg.inv(s2)

p_class1 = 40687/n
p_class2 = 104499/n
tr = p_class2/p_class1
absolute1 = np.linalg.det(s1)
absolute2 = np.linalg.det(s2)
log = np.log((absolute1/absolute2))

classifier = 2*tr + log

Scal = np.dot(mean2.T, np.dot(sigma2_inv,mean2) ) - np.dot(mean1.T,np.dot(sigma1_inv,mean1))

y_hat=[]

for val in test:
    val = np.array(val)
    val = val.reshape(3,1)
    Q = np.dot(val.T,np.dot((sigma2_inv-sigma1_inv) , val))
    L = np.dot(val.T, (np.dot(sigma2_inv,mean2) - np.dot(sigma1_inv,mean1)))
    pre = Q -L + Scal
    if pre > classifier :
        y_hat.append(1)
    else :
        y_hat.append(2)



y_hat = pd.DataFrame(y_hat)
y_hat=pd.concat([ids, y_hat],axis=1)
y_hat.to_csv('cla_test_y1.csv', encoding='utf-8')










