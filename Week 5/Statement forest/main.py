import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR
import sklearn.model_selection as mod_sel

"Размер случайного леса"

# Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
data = pd.read_csv('abalone.csv')

# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код:
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.
X = data[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]
y = data['Rings']

# Обучите случайный лес


kf5 = KFold(n_splits=5, shuffle=True, random_state=1)

for k in range(1, 51):
    rfr = RFR(random_state=1, n_estimators=k)

    scores = mod_sel.cross_val_score(X=X, y=y, estimator=rfr, cv=kf5, scoring='r2')
    avg_score = scores.mean()
    print(k, avg_score)

# Определите, при каком минимальном количестве деревьев случайный лес показывает качество на кросс-валидации выше 0.52.
