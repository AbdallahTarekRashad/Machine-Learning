import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import Callback
from keras.utils import np_utils

np.random.seed(1337)
# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128 
num_classes = 47
num_epoch = 10

# Read the train and test datasets
test_db  = pd.read_csv("emnist-balanced-test.csv")

print('test shape:', test_db.shape)

# Reshape the data to be used by a Theano CNN. Shape is
# (nb_of_samples, nb_of_color_channels, img_width, img_heigh)
X_test = test_db.iloc[:, 1:].values.reshape(test_db.shape[0],  img_rows, img_cols,1)
in_shape = ( img_rows, img_cols,1)
y_test = test_db.iloc[:, 0] 


X_test = X_test.astype('float32')
X_test /= 255

# convert class vectors to binary class matrices (ie one-hot vectors)
Y_test = np_utils.to_categorical(y_test, num_classes)

#Display the shapes to check if everything is ok

print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model = Sequential()


model.add(Convolution2D(15, 5, 5, activation = 'relu', input_shape=in_shape, init='he_normal'))


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(30, 5, 5, activation = 'relu', init='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
model.add(Flatten())
model.add(Dense(250, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(125, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax', init='he_normal')) #Last layer with one output per class


model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])
model.load_weights('CNN2Weights.h5')


print('Model built and load weights')

print( model.evaluate(X_test, Y_test, verbose = 0))