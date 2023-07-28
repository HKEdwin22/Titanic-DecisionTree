# Import libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('./Data/titanic_train.csv')

# Get known about the data
for i in df.columns:
    print(i)
print('Number of attributes: %d' %len(df.columns))
print('Number of entries: %d' %len(df))

# Check any empty targets
df['Survived'].fillna('Hello', inplace=True)
faud = df.loc[df['Survived'] == 'Hello']
print('There are %d empty targets.' %len(faud))

# Data cleansing
df.dropna(inplace=True)

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