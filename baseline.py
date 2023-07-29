# Import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

# Call, train and evaluate the model
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# prediction_train = clf.predict(X_train)
# prediction_valid = clf.predict(X_valid)

# accuracy = accuracy_score(y_train, prediction_train)
# print('The accuracy of the train set is %.4f.' %accuracy)
# accuracy = accuracy_score(y_valid, prediction_valid)
# print('The accuracy of the validation set is %.4f.' %accuracy)

# # save the model
# file = 'baseline.pickle'
# pickle.dump(clf, open(file, 'wb'))

# load the model
# load_clf = pickle.load(open('./baseline.pickle', 'rb'))

# Train the model with cross-validation
depth = []
cv = 7
for i in range(3, 21):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=4)
    depth.append((i, scores.mean()))
print(depth)

# Print the training results
ax_x = [i for i in range(3,21)]
ax_y = [i[1] for i in depth]

fig = plt.figure(figsize=(16,9))
plt.plot(ax_x, ax_y)

plt.xticks(np.array(range(3,21)))
plt.xlabel('Depth')
plt.ylabel('Accuracy on Training Set')
plt.title('%d-fold Cross Validation Training' %cv)

plt.savefig('%d-fold Cross Validation Training.jpg' %cv)
plt.show()

pass