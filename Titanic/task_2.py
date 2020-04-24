# Task 1 Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import re

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# Task 2 Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
# возраст пассажира (Age) и его пол (Sex).

tr = data[~np.isnan(data['Age'])]


# Task 3 Обратите внимание, что признак Sex имеет строковые значения.
tr.insert(0, 'Gender', data['Sex'].map({'female': 0, 'male': 1}).astype(int))

# Task 4 Выделите целевую переменную — она записана в столбце Survived.
X = tr[['Pclass', 'Fare', 'Age', 'Gender']]
Y = tr[['Survived']]

# Task 5 В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки,
# и удалите их из выборки.

# Task 6 Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию
# (речь идет о параметрах конструктора DecisionTreeСlassifier).

# Task 7 Вычислите важности признаков и найдите два признака с наибольшей важностью.
# Их названия будут ответами для данной задачи
# (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).


clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

importances = clf.feature_importances_

print(importances)

indices = np.argsort(importances)[::-1]

print(X.columns[indices])

print(X.describe())

