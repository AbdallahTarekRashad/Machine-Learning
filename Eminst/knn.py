import numpy as np 
import pandas as pd

train_db = pd.read_csv("emnist-balanced-train.csv")
test_db  = pd.read_csv("emnist-balanced-test.csv")
y_train = train_db.iloc[:,0]
x_train = train_db.iloc[:,1:]
y_test = test_db.iloc[:,0]
x_test = test_db.iloc[:,1:]


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train) 
print("fit data!!!!!!!!!!!!!")
print(neigh.score(x_test,y_test))