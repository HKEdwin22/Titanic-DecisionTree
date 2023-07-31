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

# Split the dataset into training and testing set
# X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=1)

'''
Call, train and evaluate the model
'''
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# prediction_train = clf.predict(X_train)
# prediction_valid = clf.predict(X_valid)

# accuracy = accuracy_score(y_train, prediction_train)
# print('The accuracy of the train set is %.4f.' %accuracy)
# accuracy = accuracy_score(y_valid, prediction_valid)
# print('The accuracy of the validation set is %.4f.' %accuracy)

# # Plot the confusion matrix
# cm = confusion_matrix(y_valid, prediction_valid)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Prediction Results with GSMOTE Applied / %.4f' %accuracy)
# plt.xlabel('Ground Truth')
# plt.ylabel('Prediction')
# plt.show()


# save the model
# file = 'baseline.pickle'
# pickle.dump(clf, open(file, 'wb'))

# load the model
# load_clf = pickle.load(open('./baseline.pickle', 'rb'))


'''
Find out the level of depth without overfitting
'''
cv = 5
scr_tr, scr_val = [], []
for i in range(1, 15):
    clf = DecisionTreeClassifier(max_depth=i, random_state=0)
    scores = cross_validate(estimator=clf, X=X_resampled, y=y_resampled, cv=cv, n_jobs=4, return_train_score=True)
    scr_tr.append(scores['train_score'].mean())
    scr_val.append(scores['test_score'].mean())

# Print the training results
ax_x = [i for i in range(1,15)]

fig = plt.figure(figsize=(16,9))
plt.plot(ax_x, scr_tr, color='#6a79a7')
plt.plot(ax_x, scr_val, color='#d767ad')
fig.legend(['Training', 'Validation'])

plt.annotate(float(round(max(scr_val), 4)), xy=(scr_val.index(max(scr_val))+.6,max(scr_val)+.001))
plt.xticks(np.array(range(1,15)))
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Performance in Training and Validation with %d-fold Cross Validation' %cv)

plt.savefig('GSMOTE_best depth search.png')
plt.show()


'''
Feature Selection
'''
# Show correlation between the variables and target
# fig, ax = plt.subplots(figsize=(14,14))
# sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap='RdBu')
# plt.title('Correlation between attributes and the target')
# plt.savefig('Correlation.jpg')
# plt.show()



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