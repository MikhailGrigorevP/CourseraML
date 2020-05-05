import numpy as np
import os
from pandas import read_csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt

"Градиентный бустинг над решающими деревьями"

# Загрузите выборку из файла gbm-data.csv с помощью pandas и
# преобразуйте ее в массив numpy (параметр values у датафрейма)
data = read_csv('gbm-data.csv')

# В первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер, форма и т.д.
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

if not os.path.exists('plots'):
    os.makedirs('plots')


def plot(train_loss, test_loss, fname):
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig(fname)


# Обучите GradientBoostingClassifier
# Используйте метод staged_decision_function для предсказания качества
# на обучающей и тестовой выборке на каждой итерации.
# Преобразуйте полученное предсказание с помощью сигмоидной
min_losses = {}
for index, learning_rate in enumerate([1, 0.5, 0.3, 0.2, 0.1], start=1):
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    train_pred_iters = clf.staged_predict_proba(X_train)
    test_pred_iters = clf.staged_predict_proba(X_test)
    train_loss = [log_loss(y_train, pred) for pred in train_pred_iters]
    test_loss = [log_loss(y_test, pred) for pred in test_pred_iters]
    best_iter = np.argmin(test_loss)
    min_losses[learning_rate] = (test_loss[best_iter], best_iter)
    plot(train_loss, test_loss, 'plots/%d_%.1f.png' % (index, learning_rate))

# Вычислите и постройте график значений log-loss

# Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации: переобучение (
# overfitting) или недообучение (underfitting)?
with open('q1.txt', 'w') as output:
    output.write('overfitting')

# Приведите минимальное значение log-loss на тестовой выборке и номер итерации,
# на котором оно достигается, при learning_rate = 0.2
with open('q2.txt', 'w') as output:
    output.write('%.2f %d' % min_losses[0.2])

# На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
# на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта
rf = RandomForestClassifier(n_estimators=min_losses[0.2][1], random_state=241)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_score = log_loss(y_test, rf_pred)

with open('q3.txt', 'w') as output:
    output.write('%.2f' % (rf_score))
