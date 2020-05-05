from pandas import read_csv
from numpy import corrcoef, argmax
from plotly.matplotlylib.mplexporter._py3k_compat import reduce
from sklearn.decomposition import PCA

"Составление фондового индекса"

# Загрузите данные close_prices.csv.
# В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.

# На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты

prices = read_csv('close_prices.csv').iloc[:, 1:]

pca = PCA(n_components=10)
pca.fit(prices)
pc = pca.transform(prices)[:, 0]

count = reduce(lambda acc, i: (acc[0] + 1, acc[1] + i) if acc[1] < 0.9 else acc,
               pca.explained_variance_ratio_, (0, 0))[0]

with open("q1.txt", "w") as output:
    output.write('%d' % (count))

# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса

dj = read_csv('djia_index.csv')['^DJI'].values
correlation = corrcoef(pc, dj)[0, 1]


with open("q2.txt", "w") as output:
    output.write('%.2f' % correlation)

most_influential_company = prices.columns[argmax(pca.components_[0])]

# Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.

with open("q3.txt", "w") as output:
    output.write('%s' % (most_influential_company))
