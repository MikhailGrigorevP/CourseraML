import pandas
import numpy as np
from sklearn.svm import SVC

"Опорные объекты"

# Загрузите выборку из файла svm-data.csv.
# В нем записана двумерная выборка (целевая переменная указана в первом столбце, признаки — во втором и третьем)
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

# Обучите классификатор с линейным ядром

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(x, y)
predictions = clf.predict(x)

# Найдите номера объектов, которые являются опорными (нумерация с единицы)

print(predictions)
print(clf.support_)
