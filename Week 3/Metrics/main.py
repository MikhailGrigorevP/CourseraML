import pandas
import numpy as np
import sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,\
    roc_auc_score, precision_recall_curve

"Метрики качества классификации"

# Загрузите файл classification.csv.
# В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred)

data = pandas.read_csv('classification.csv')

t = data[data['true'] == data['pred']]['pred']
tp = t.sum()
tn = t.size - tp

f = data[data['true'] != data['pred']]['pred']
fp = f.sum()
fn = f.size - fp

# Заполните таблицу ошибок классификации
# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.

with open("q1.txt", "w") as output:
    output.write('%d %d %d %d' % (tp, fp, fn, tn))

# Посчитайте основные метрики качества классификатора:

accuracy = accuracy_score(data['true'], data['pred'])
precision = precision_score(data['true'], data['pred'])
recall = recall_score(data['true'], data['pred'])
F = f1_score(data['true'], data['pred'])

with open("q2.txt", "w") as output:
    output.write('%.2f %.2f %.2f %.2f' % (accuracy, precision, recall, F))

# Имеется четыре обученных классификатора.
# В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу
# для каждого классификатора на некоторой выборке

scores = pandas.read_csv('scores.csv')

algs = scores.columns[1:]

# Посчитайте площадь под ROC-кривой для каждого классификатора


def find_best_alg(score_func):
    s = algs.map(lambda alg: [score_func(alg), alg])

    return np.sort(s)[::-1][0]


best_roc, best_roc_alg = find_best_alg(lambda alg:
                                       roc_auc_score(scores['true'], scores[alg]))

with open("q3.txt", "w") as output:
    output.write('%s' % (best_roc_alg))

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70%


def best_prc_score(alg):
    prc = precision_recall_curve(scores['true'], scores[alg])
    fr = pandas.DataFrame({'precision': prc[0], 'recall': prc[1]})
    return fr[fr['recall'] >= 0.7]['precision'].max()


best_prc, best_prc_alg = find_best_alg(best_prc_score)

with open("q4.txt", "w") as output:
    output.write('%s' % (best_prc_alg))
