# Import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from gsmote import GeometricSMOTE

# Read data
df = pd.read_csv('./Data/titanic_train.csv')

'''
Data Preprocessing
'''
# Drop irrelevant attributes
df = df.drop(['Unnamed: 0', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Count total missing values for each attribute
dummy = df.isnull().sum()

# Drop unused attributes
df = df.drop(['Cabin'], axis=1)

# Check any invalid format
def check_format(a, type):
    if type == 'num':
        dummy = df.applymap(lambda x: isinstance(x, (int, float)))[a]
    elif type == 'str':
        dummy = df.applymap(lambda x: isinstance(x, (str)))[a]
    dummy = dummy.to_frame()
    dummy = dummy[a].value_counts()
    print(dummy)
    
num_att = ['Age', 'Parch', 'Pclass', 'SibSp', 'Family_Size', 'Fare']
str_att = ['Embarked', 'Sex', 'Title']

for i in num_att:
    check_format(i, 'num')
for i in str_att:
    check_format(i, 'str')

# Encode categorical data
le = preprocessing.LabelEncoder()
df['Gender'] = le.fit_transform(df['Sex'])
df['Port'] = le.fit_transform(df['Embarked'])
df['Title_factor'] = le.fit_transform(df['Title'])

# Data cleansing 1 - Drop all null data
df.dropna(inplace=True)

# Data cleansing 2 - Replace null data with mode/mean/median

# Extract attributes and target
X = df.drop(['Survived', 'Sex', 'Embarked', 'Title'], axis=1)
y = df['Survived']

# Oversampling
Gsmote = GeometricSMOTE()
X_resampled, y_resampled = Gsmote.fit_resample(X, y)
X_resampled = X_resampled[['Title_factor', 'Gender']]

# Split the dataset into training and testing set
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=1)



'''
Grid Search
'''
best_acc = 0

clf = DecisionTreeClassifier(random_state=0)
grid = {'max_depth': [i for i in range(1,21)], 'min_samples_leaf': [i for i in range(50,801,50)], 'min_samples_split': [i for i in range(50,801,50)]}
for g in ParameterGrid(grid):
    clf.set_params(**g)
    clf.fit(X_train, y_train)
    scr_tr = accuracy_score(y_train, clf.predict(X_train))
    scr_val = accuracy_score(y_valid, clf.predict(X_valid))
    if scr_val >= best_acc:
        best_acc = scr_val
        best_grid = g
        best_scr_tr = scr_tr

print('Best Grid: ', best_grid)
print('validation accuracy: %.4f         training accuracy: %.4f' %(best_acc, best_scr_tr))
        


pass