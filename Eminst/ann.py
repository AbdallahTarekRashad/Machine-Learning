import pandas as pd
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout
from keras.models import Model
num_classes = 47

inp = Input(shape=(784,))
hidden_1 = Dense(1024, activation='relu')(inp)
dropout_1 = Dropout(0.2)(hidden_1)
out = Dense(num_classes, activation='softmax')(hidden_1) 
model = Model(input=inp, output=out)
test_db  = pd.read_csv("emnist-balanced-test.csv")

model.load_weights('ANNWeights.h5')

print('Model built and load weights')

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy']) 

y_test = test_db.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)
print ("y_test:", y_test.shape)

x_test = test_db.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print(model.evaluate(x_test, y_test, verbose=1)) 

