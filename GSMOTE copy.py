# Import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
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
df = df.drop(['Unnamed: 0', 'Ticket', 'Name', 'PassengerId', 'Parch', 'SibSp'], axis=1)

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
    
num_att = ['Age', 'Pclass', 'Family_Size', 'Fare']
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

# Create a new feature Fare Per Head
df['FarePerHead'] = (df['Fare']/df['Family_Size']).where(df['Family_Size']>0, df['Fare'])

# Data cleansing 1 - Drop all null data
df.dropna(inplace=True)

# Extract attributes and target
X = df.drop(['Survived', 'Sex', 'Embarked', 'Title', 'Fare'], axis=1)
y = df['Survived']

# Oversampling
Gsmote = GeometricSMOTE()
X_resampled, y_resampled = Gsmote.fit_resample(X, y)
X_resampled = X_resampled[['Title_factor', 'Gender', 'FarePerHead']]
# X_resampled = X_resampled.drop(['Port'], axis=1)

# Split the dataset into training and testing set
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=1)



'''
Call, train and evaluate the model
'''
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# prediction_train = clf.predict(X_train)
# prediction_valid = clf.predict(X_valid)

# scr_tr = accuracy_score(y_train, prediction_train)
# print('The accuracy of the train set is %.4f.' %scr_tr)
# scr_val = accuracy_score(y_valid, prediction_valid)
# print('The accuracy of the validation set is %.4f.' %scr_val)

# # # Plot the confusion matrix
# cm = confusion_matrix(y_valid, prediction_valid)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Prediction Results with GSMOTE and Top 3 Features / (%.4f / %.4f)' %(scr_tr, scr_val))
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.savefig('Prediction_GSMOTE_3 Features.png')
# plt.show()



'''
Train and validate the model with 4 levels of depth
'''
score_tr, score_val, cm = [], [], []

for i in range(1, 21):
    clf = DecisionTreeClassifier(max_depth=i, random_state=0)
    clf.fit(X_train, y_train)

    pred_tr = clf.predict(X_train)
    score_tr.append(accuracy_score(y_train, pred_tr))

    pred_valid = clf.predict(X_valid)
    score_val.append(accuracy_score(y_valid, pred_valid))

    # Confusion Matrix
    cm.append(confusion_matrix(y_valid, clf.predict(X_valid)))

fig, axn = plt.subplots(nrows=1, ncols=3, figsize=(16,9))
for i, ax in enumerate(axn.flat):
    sns.heatmap(cm[i-1], ax=ax, annot=True, fmt='d', cmap='Blues', cbar=i==2)
    ax.set_title('%d level / %.4f' %(i+1, score_val[i]))
fig.suptitle('Prediction Results')
plt.savefig('Confusion Matrix.jpg')
plt.show()

# print(score_val)
# Lv 1 = 0.7765, Lv 2 = 0.7430 Lv3 = 0.7709


# Plot the accuracy
ax_x = [i for i in range(1, len(score_tr)+1)]
fig = plt.figure(figsize=(12,9))
plt.plot(ax_x, score_tr, color="#6a79a7")
plt.plot(ax_x, score_val, color='#d767ad')

fig.legend(['Training', 'Validation'])
plt.title('Accuracy on Training and Validation Set')
plt.xticks(np.array(ax_x))
plt.xlabel('Depth')
plt.ylabel('Accuracy')

plt.savefig('Overfitting Pattern with GSMOTE-MI models.jpg')
plt.show()



pass