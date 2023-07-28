# Import libraries
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('./Data/titanic_train.csv')

# Check any empty targets
df['Survived'].fillna('Hello', inplace=True)
faud = df.loc[df['Survived'] == 'Hello']
print('There are %d empty targets.' %faud)

# Extract targets and independent variables
X = df.loc[:, df.columns != 'Survived']
y = df.loc[:, df.columns == 'Survived']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Call and train the model
model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)

# Get known about the data
for i in df.columns:
    print(i)
print('There are %d attributes.' %len(df.columns))


pass