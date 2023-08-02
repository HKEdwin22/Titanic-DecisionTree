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

# Load classifier
file = './final_3 features.pickle'
load_clf = pickle.load(open(file, 'rb'))

