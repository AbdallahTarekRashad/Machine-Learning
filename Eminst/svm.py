from keras.models import Sequential
from keras.utils import np_utils
import pandas as pd
import numpy as np

emnist_test = pd.read_csv('emnist-balanced-test.csv',header = None)


y_test = emnist_test.loc[:,0]
x_test = emnist_test.loc[:,1:785]

x_test = np.divide(x_test,255)
x_test = np.array(x_test)


batch_size = 100
nr_classes = 47
nr_iterations = 100

Y_test = np_utils.to_categorical(y_test, nr_classes)



model = Sequential()
model.add(Dense(nr_classes, input_shape=(784,)))

model.compile(loss='hinge',
              optimizer='sgd',
              metrics=['accuracy'])

model.load_weights('SVMWeights.h5')



print('Model built and load weights')

print( model.evaluate(x_test, Y_test, verbose = 0))

