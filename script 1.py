# Import libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
df = df.loc[:, df.columns != 'Name']
df = df.loc[:, df.columns != 'Passenger ID']

# Count total missing values for each attribute
dummy = df.isnull().sum()

# Drop unused attribute
df = df.loc[:, df.columns != 'Cabin']

# Check any invalid format
def check_format(x, type):
    if type == 'num':
        dummy = df.applymap(lambda x: isinstance(x, (int, float)))[i]
    elif type == 'str':
        dummy = df.applymap(lambda x: isinstance(x, (str)))[i]
    dummy = dummy.to_frame()
    dummy = dummy[i].value_counts()
    print(dummy)
    
num_att = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
str_att = ['Embarked', 'Sex', 'Ticket', 'Title']

for i in num_att:
    check_format(i, 'num')
for i in str_att:
    check_format(i, 'str')
    
# Data cleansing 1 - Drop all null data
df.dropna(inplace=True)

# Data cleansing 2 - Replace null data with mode/mean/median

# Extract targets and independent variables
X = df.loc[:, df.columns != 'Survived']
y = df.loc[:, df.columns == 'Survived']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Call, train and evaluate the model
model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test, y_test)

accuracy = accuracy_score(y_test,y_pred)
print('The prediction accuracy is %d.' %accuracy)





pass