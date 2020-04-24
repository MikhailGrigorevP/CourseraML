import pandas
import re

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# print(data)

# Task 1 Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел
sex_counts = data['Sex'].value_counts()
print(1, '{} {}'.format(sex_counts['male'], sex_counts['female']))

# Task 2 Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
survived_counts = data['Survived'].value_counts()
survived_percent = 100.0 * survived_counts[1] / survived_counts.sum()
print(2, "{:0.2f}".format(survived_percent))

# Task 3 Какую долю пассажиры первого класса составляли среди всех пассажиров?
pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print(3, "{:0.2f}".format(pclass_percent))

# Task 4 Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
ages = data['Age'].dropna()
print(4, "{:0.2f} {:0.2f}".format(ages.mean(), ages.median()))

# Task 5 Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
corr = data['SibSp'].corr(data['Parch'])
print(5, "{:0.2f}".format(corr))


# Task 6 Какое самое популярное женское имя на корабле?

def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')
    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()

print(6, name_counts.head(1).index.values[0])
