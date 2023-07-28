# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('./Data/titanic_train.csv')

# Get known about the data
print(df.head())

for i in df.columns:
    print(i)
print('Number of attributes: %d' %len(df.columns))
print('Number of entries: %d' %len(df))

'''
Data Preprocessing
'''
# Drop irrelevant attributes
df = df.drop(['Ticket', 'Name', 'PassengerId', 'Fare'], axis=1)

# Count total missing values for each attribute
dummy = df.isnull().sum()

# Drop unused attributes
df = df.drop(['Cabin'], axis=1)

# Check any invalid format
def check_format(x, type):
    if type == 'num':
        dummy = df.applymap(lambda x: isinstance(x, (int, float)))[i]
    elif type == 'str':
        dummy = df.applymap(lambda x: isinstance(x, (str)))[i]
    dummy = dummy.to_frame()
    dummy = dummy[i].value_counts()
    print(dummy)
    
num_att = ['Age', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
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

# Extract targets and independent variables
X = df.drop(['Sex', 'Survived', 'Embarked', 'Title'], axis=1)
y = df['Survived']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Call, train and evaluate the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print('The prediction accuracy is %.4f.' %accuracy)





pass