

import numpy as np
feautes=[]
labels=[]

features = np.array([[0.1,0.1],
		[0.1,1.0],
		[1.0,0.1],
		[1.0,1.0]
		])
labels = np.array([0 ,1, 1, 0])

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = SVM()
model = gnb.fit(features, labels)
y_pred = model.predict(features)
print(y_pred)