import pandas
from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

"Анализ текстов"

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

X = newsgroups.data
y = newsgroups.target

# Вычислите TF-IDF-признаки для всех текстов.
# Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным.

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для
# SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам.

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)

# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге

clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

score = 0
C = 0

for i in range(11):
    if gs.cv_results_['mean_test_score'][i] > score:
        score = gs.cv_results_['mean_test_score'][i]
        C = gs.cv_results_['params'][i]['C']

model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(X), y)

# Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC)

words = vectorizer.get_feature_names()
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
top_words = top_words.sort_values()
print(1, ','.join(top_words))
