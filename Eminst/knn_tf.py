import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils

emnist_train = pd.read_csv('emnist-balanced-train.csv',header = None)
emnist_test = pd.read_csv('emnist-balanced-test.csv',header = None)

y_train = emnist_train.loc[:,0]
x_train = emnist_train.loc[:,1:785]
y_test = emnist_test.loc[:,0]
x_test = emnist_test.loc[:,1:785]

x_train = np.divide(x_train,255)
x_test = np.divide(x_test,255)
x_train = np.matrix(x_train)
x_test = np.matrix(x_test)
x_train = x_train.astype(float)
x_test = x_test.astype(float)


y_train = np.array(y_train)
y_train = y_train.astype(float)
y_train = y_train.reshape(112800,1)

y_test = np.array(y_test)
y_test = y_test.astype(float)
y_train = np_utils.to_categorical(y_train, 47)
y_test = np_utils.to_categorical(y_test, 47)

# tf Graph Input
xtr = tf.placeholder(dtype="float64", shape=[None,784])
ytr = tf.placeholder(dtype="float64", shape=[None,47])
xte = tf.placeholder(dtype="float64", shape=[1,784])


#K-near
K=5 #how many neighbors
nearest_neighbors=tf.Variable(tf.zeros([K]))

#model
distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(xtr, xte)),axis=1)) #L1
# the negitive above if so that top_k can get the lowest distance *_* its a really good hack i learned
values,indices=tf.nn.top_k(distance,k=K,sorted=False)

#a normal list to save
nn = []
for i in range(K):
    nn.append(tf.argmax(ytr[indices[i]], 0)) #taking the result indexes

#saving list in tensor variable
nearest_neighbors=nn
# this will return the unique neighbors the count will return the most common's index
y, idx, count = tf.unique_with_counts(nearest_neighbors)

pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
# this is tricky count returns the number of repetation in each elements of y and then by begining from that and size begin 1
# it only returns that neighbors value : for example
# suppose a is array([11,  1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  4,  4,  4,  4]) so unique_with_counts of a will
#return y= (array([ 1,  2,  3,  4, 11]) count= array([3, 3, 2, 7, 1])) so argmax of count will be 3 which will be the
#index of 4 in y which is the hight number in a

#setting accuracy as 0
accuracy=0

#initialize of all variables
init=tf.global_variables_initializer()

#start of tensor session
with tf.Session() as sess:

    for i in range(x_test.shape[0]):
        #return the predicted value
        predicted_value=sess.run(pred, feed_dict={xtr: x_train , ytr: y_train_cat,xte: x_test[i, :]})

        print("Test",i,"Prediction",predicted_value,"True Class:",np.argmax(y_test[i]))

        if predicted_value == np.argmax(y_test[i]):
            # if the prediction is right then a double value of 1./200 is added 200 here is the number of test
                accuracy += 1. / len(x_test)
              
    print("Calculation completed ! ! ")
print(K,"-th neighbors' Accuracy is:",accuracy)