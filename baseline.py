# Import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

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
df = df.drop(['Unnamed: 0', 'Ticket', 'Name', 'PassengerId', 'Fare'], axis=1)

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


'''
Baseline model
'''
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


'''
Feature Selection
'''
# best_f = []
# score = []
# MI = mutual_info_classif(X, y, random_state=0)
# for scr, feature in sorted (zip(MI, X.columns), reverse=True):
#     print(feature, round(scr, 4))
#     score.append(scr)
#     best_f.append(feature)

# fig = plt.figure(figsize=(12,9))
# plt.scatter(best_f, score)

# plt.title('MI of Features')
# plt.xlabel('Feature')
# plt.ylabel('Mutual Information Score')

# plt.savefig('MI of Attributes.png')
# plt.show()


'''
Build the model with feature selection
'''
X = X[['Title_factor', 'Gender']]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
pred_valid = clf.predict(X_valid)

scr_tr = accuracy_score(y_train, clf.predict(X_train))
print(scr_tr)

acc_scr = accuracy_score(y_valid, pred_valid)
cm = confusion_matrix(y_valid, pred_valid)

fig = plt.figure(figsize=(12,9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Prediction with Feature Selection / %.4f' %acc_scr)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')

plt.savefig('Confusion Matrix for model 2.png')
plt.show()




# Show correlation between the variables and target
# fig, ax = plt.subplots(figsize=(14,14))
# sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap='RdBu')
# plt.title('Correlation between attributes and the target')
# plt.savefig('Correlation.jpg')
# plt.show()




'''
Find out the level of depth without overfitting
'''
scr_tr, scr_val = [], []
for i in range(1, 15):
    clf = DecisionTreeClassifier(max_depth=i, random_state=0)
    clf.fit(X_train, y_train)

    pred_tr = clf.predict(X_train)
    scr_tr.append(accuracy_score(y_train, pred_tr))

    pred_valid = clf.predict(X_valid)
    scr_val.append(accuracy_score(y_valid, pred_valid))




# 5-fold Cross Validation
# cv = 5
# scr_tr, scr_val = [], []
# for i in range(1, 15):
#     clf = DecisionTreeClassifier(max_depth=i, random_state=0)
#     scores = cross_validate(estimator=clf, X=X, y=y, cv=cv, n_jobs=4, return_train_score=True)
#     scr_tr.append(scores['train_score'].mean())
#     scr_val.append(scores['test_score'].mean())

# # Print the training results
ax_x = [i for i in range(1,15)]

fig = plt.figure(figsize=(16,9))
plt.plot(ax_x, scr_tr, color='#6a79a7')
plt.plot(ax_x, scr_val, color='#d767ad')
fig.legend(['Training', 'Validation'])

plt.xticks(np.array(range(1,15)))
plt.xlabel('Depth')
plt.ylabel('Accuracy on Training Set')
plt.title('Performance in Training and Validation')

plt.savefig('baseline.png')
plt.show()



'''
Train and validate the model with 4 levels of depth
'''
score_tr, score_val, cm = [], [], []

for i in range(1, 4):
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

print(score_val)
# Lv 1 = 0.7765, Lv 2 = 0.7430 Lv3 = 0.7709


# Plot the accuracy
ax_x = [i for i in range(1,4)]
fig = plt.figure(figsize=(12,9))
plt.plot(ax_x, score_tr, color="#6a79a7")
plt.plot(ax_x, score_val, color='#d767ad')

fig.legend(['Training', 'Validation'])
plt.title('Accuracy on Training and Validation Set')
plt.xticks(np.array(range(1,4)))
plt.xlabel('Depth')
plt.ylabel('Accuracy')

plt.savefig('overfitting 2.jpg')
plt.show()




'''
Train the model with cross-validation
'''
depth = []
cv = 7
for i in range(3, 21):
    clf = DecisionTreeClassifier(max_depth=i, random_state=0)
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=3)
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

# plt.savefig('%d-fold Cross Validation Training.jpg' %cv)
plt.show()






pass