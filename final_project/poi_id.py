#!/usr/bin/python3

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# By using ExtraTreesClassifier from sklearn we identified 3 features that were
# not contributing much for the model:
#                   loan_advances, director_fees, restricted_stock_deferred
# So we decided to remove these features from the analysis.
np.random.seed(1234)

features_list = ['deferred_income',
                 'expenses',
                 'exercised_stock_options',
                 'restricted_stock',
                 'from_this_person_to_poi',
                 'salary',
                 'total_stock_value',
                 'bonus',
                 'shared_receipt_with_poi',
                 'other',
                 'from_poi_to_this_person',
                 'long_term_incentive',
                 'total_payments',
                 'from_messages',
                 'to_messages',
                 'deferral_payments',
                 'poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# Generate Dataframe out of the data_dict
df_enron = pd.DataFrame.from_dict(data_dict, orient='index')


# Define columns of features
# df_enron.drop('email_address', axis=1, inplace=True)
df_enron = df_enron[features_list]

# Change columns type to numeric
df_enron = df_enron.apply(pd.to_numeric, errors='coerce')

# Fillna with 0 to replace NaN
df_enron = df_enron.fillna(0.0)

# Rescaling feature values with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(df_enron.drop('poi', axis=1))
y = df_enron.poi

# Define X and y
# X = df_enron.drop('poi', axis=1).values
# y = df_enron['poi'].values

# Using DBSCAN to spot outliers
from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
    eps=0.5,
    metric="euclidean",
    min_samples=3,
    n_jobs=-1)
clusters = outlier_detection.fit_predict(X)
df_dbscan = pd.DataFrame(clusters, index=df_enron.index)
#print(df_dbscan[df_dbscan[0] == -1].index)

list_dbscan_outliers = list(df_dbscan[df_dbscan[0] == -1].index)
print(list_dbscan_outliers)

# As mentioned on the report, we applied PCA to define outliers and ended
# finding one sample called "TOTAL" as a candidate outlier
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
dim_red = pca.fit_transform(X)
# print(pca.singular_values_)

# Transform PCA results into Dataframe
pca_df = pd.DataFrame(dim_red, columns=[
    'PCA1', 'PCA2'], index=df_enron.index)

# Concatenate with y series as for labels
pca_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)

# Get z-score for each component
pca_df['PCA1_zscore'] = (pca_df.PCA1 - pca_df.PCA1.mean()) / pca_df.PCA1.std()
pca_df['PCA2_zscore'] = (pca_df.PCA2 - pca_df.PCA2.mean()) / pca_df.PCA2.std()


# Get rows that are 3 standard deviations away from mean
list_outliers_pca = (pca_df[(np.abs(pca_df.PCA1_zscore) > 3)
                            | (np.abs(pca_df.PCA2_zscore) > 3)]).index

# print(list_outliers_pca)

"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_features_no_outliers = df_enron.drop(list_outliers_pca)
X = scaler.fit_transform(df_features_no_outliers.drop('poi', axis=1))
y = df_features_no_outliers.poi
"""

# Spliting data into trainning and testing
from sklearn.model_selection import train_test_split
# Split your data X and y, into a training and a test set and fit the
# pipeline onto the training data
# X_train, X_test, y_train, y_test = train_test_split(None
#    X_resampled,  y_resampled, test_size=0.3,
#    random_state=0)

from imblearn.over_sampling import SMOTE
# Define the resampling method
method = SMOTE(kind='regular', random_state=42)

# Create the resampled feature set
X_resampled, y_resampled = method.fit_sample(X, y)


from sklearn.utils import shuffle
X_resampled,  y_resampled = shuffle(X_resampled,  y_resampled)


# Split your data X and y, into a training and a test set and fit the
# pipeline onto the training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,  y_resampled, test_size=0.5, random_state=0)


# print(df_enron)
# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                             oob_score=False, random_state=42, verbose=0,
                             warm_start=False)
print(clf)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
probs = clf.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
print(roc_auc_score(y_test, probs[:, 1]))
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))


# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.model_selection import StratifiedShuffleSplit
"""
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X_resampled, y_resampled):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                                 oob_score=False, random_state=None, verbose=0,
                                 warm_start=False)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    # Print the ROC curve, classification report and confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    print(roc_auc_score(y_test, probs[:, 1]))
    print(classification_report(y_test, predicted))
    print(confusion_matrix(y_test, predicted))
"""
# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
