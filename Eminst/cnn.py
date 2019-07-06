from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
import pandas as pd

test = pd.read_csv("emnist-balanced-test.csv")
def load_char_mappings(mapping_path):
    """
    load EMNIST character mappings. This maps a label to the correspondent byte value of the given character
    return: the dictionary of label mappings
    """
    mappings = {}
    with open(mapping_path) as f:
        for line in f:
            (key, val) = line.split()
            mappings[int(key)] = int(val)

    return mappings
def rotate(img):
    flipped = np.fliplr(img.reshape(28,28))
    return np.rot90(flipped).reshape(784,)
        



num_classes = 47


y_test = test.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)
print ("y_test:", y_test.shape)

x_test = test.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print ("x_test:",x_test.shape)

x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
print(x_test.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('CNNmodel.h5')

print('Model built and load weights')


print(model.evaluate(x_test, y_test, verbose=1)) 
