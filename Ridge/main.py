def load_data():
    import pandas
    return pandas.read_csv('salary-train.csv'), pandas.read_csv('salary-test-mini.csv')

def normalize_data(data):
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()

data_train, data_test = load_data()

normalize_data(data_train)
normalize_data(data_test)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5)
X_train_words = tfidf.fit_transform(data_train['FullDescription'])
X_test_words = tfidf.transform(data_test['FullDescription'])

from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

from scipy.sparse import hstack
X_train = hstack([X_train_words, X_train_categ])
X_test = hstack([X_test_words, X_test_categ])

y_train = data_train['SalaryNormalized']

from sklearn.linear_model import Ridge
model = Ridge(alpha=1)
model.fit(X_train, y_train)

result = model.predict(X_test)

with open("q1.txt", "w") as output:
    output.write(' '.join(['%.2f' % (i) for i in result]))