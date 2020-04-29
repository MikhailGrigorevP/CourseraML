import pandas
import numpy as np
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None, names=['a', 'b', 'c'])

y = []
for i in range(len(data)):
    y.append(data.a[i])
y = np.array(y)

x = np.zeros(shape=(len(data), 2))
for i in range(len(x)):
    x[i][0] = data.b[i]
    x[i][1] = data.c[i]
x = np.array(x)

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(x, y)
predictions = clf.predict(x)
print(predictions)
print(clf.support_)
