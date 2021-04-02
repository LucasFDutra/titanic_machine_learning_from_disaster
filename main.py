#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data
#%%
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)

def define_embarked_group(e):
    if e == 'S':
        return 1
    elif e == 'C':
        return 2
    return 3

train_data['Embarked'] = train_data['Embarked'].apply(lambda x: define_embarked_group(x))
train_data['Embarked']

#%%
train_data.info()

#%%
train_survived = train_data[train_data['Survived'] == 1]
train_died = train_data[train_data['Survived'] == 0]

#%%
## Baixa renda tem mair chances de morrer
train_survived.hist(column='Pclass', figsize=(15,6))
train_died.hist(column='Pclass', figsize=(15,6))

#%%
## Mais idosos tem mais chances de morrer
## crianças tem mais chances de viver
train_survived.hist(column='Age', figsize=(15,6))
train_died.hist(column='Age', figsize=(15,6))

#%%
## Homens tem mais chance de morrer

train_survived.hist(column='Sex', figsize=(15,6))
train_died.hist(column='Sex', figsize=(15,6))

#%%
## não ajuda muito

train_survived.hist(column='Embarked', figsize=(15,6))
train_died.hist(column='Embarked', figsize=(15,6))

#%%
## não ajuda muito

train_survived.hist(column='SibSp', figsize=(15,6))
train_died.hist(column='SibSp', figsize=(15,6))

#%%
## não ajuda muito

train_survived.hist(column='Parch', figsize=(15,6))
train_died.hist(column='Parch', figsize=(15,6))

#%%
## não ajuda muito

train_survived.hist(column='Fare', figsize=(15,6))
train_died.hist(column='Fare', figsize=(15,6))

#%%

## Vou dividir em 3 grupos de idades: crianças 1, jovens 2, idosos 3
# criança <= 13
# idoso >= 60

def define_age_group(age):
    if age <= 13:
        return 1
    elif age >= 60:
        return 3
    return 2

train_data['Age'] = train_data['Age'].apply(lambda x: define_age_group(x))

#%%
## elementos que mandam

# Pclass
# Age
# Sex

train_data = train_data.loc[:, ['Pclass', 'Age', 'Sex', 'Survived']]
train_data
#%%
train_data = pd.get_dummies(train_data, columns=['Pclass', 'Age', 'Sex'])
train_data

#%%
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_data.iloc[:,1:], train_data.iloc[:,0:1])
clf.score(train_data.iloc[:,1:], train_data.iloc[:,0:1])
