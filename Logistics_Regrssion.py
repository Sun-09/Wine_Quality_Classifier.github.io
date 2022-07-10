from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

car = datasets.load_wine()

x = car.data

y = (car.target == 2).astype(np.int)


clf = LogisticRegression()

clf.fit(x,y)

predict = clf.predict(([[14.13, 4.1, 2.74, 24.5, 96.00,  2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560.00]]))

if predict == 2:
    print("Not Good wine")
else:
    print("Good Wine")
