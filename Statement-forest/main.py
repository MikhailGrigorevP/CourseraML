from builtins import range
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor as RFR
import sklearn.model_selection as mod_sel
from sklearn.metrics import r2_score as R2

data = pd.read_csv('abalone.csv');

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]
y = data['Rings']


def get_r2_score(estimator, x, y_true):
    y_pred = estimator.predict(x)
    return R2(y_true, y_pred)


kf5 = KFold(n_splits=5, shuffle=True, random_state=1)

for k in range(1, 51):
    rfr = RFR(random_state=1, n_estimators=k)
    # rfr.fit(X, y)

    scores = mod_sel.cross_val_score(X=X, y=y, estimator=rfr, cv=kf5, scoring='r2')
    # scores = mod_sel.cross_val_score(X=X, y=y, estimator=rfr, cv=kf5, scoring=get_r2_score)
    avg_score = scores.mean()
    print(k, avg_score)
