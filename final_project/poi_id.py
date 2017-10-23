#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#all features
#features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
#                 'exercised_stock_options', 'bonus', 'restricted_stock',
#                 'shared_receipt_with_poi', 'restricted_stock_deferred',
#                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
#                 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
#                 'long_term_incentive', 'from_poi_to_this_person']

# features after selection (poi is a label =p)
#tirei o salario e n adiantou nada
features_list = ['poi','expenses','loan_advances', 'long_term_incentive','restricted_stock_deferred','bonus']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


keys = data_dict.keys()

#get all features again
#feature_all = data_dict[keys[0]].keys()
# get the same new features that i got above
#feature_all = ['poi','bonus','restricted_stock_deferred','expenses','loan_advances', 'long_term_incentive']
feature_all = ['poi','expenses','loan_advances', 'long_term_incentive','restricted_stock_deferred', 'bonus']

del feature_all[feature_all.index('poi')]
#del feature_all[feature_all.index('email_address')]

count_pois = 0
for key in keys:
    #print data_dict[key]['poi']
    #if data_dict[key]

    if data_dict[key]['poi'] == True:
        count_pois += 1

print "Total of people (DP): " + str(len(keys))
print "Total of POIs: " + str(count_pois)
print "Total of Non POIs: " + str(len(keys) - count_pois)
print "POI percentage: ", str(round(float(count_pois)/float(len(keys)) * 100,2)) +"%"

#prints all features
#print feature_all

### Task 2: Remove outliers
#removing crazy outlier
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



##Christian doing a little lasso just to see what it gets to us :D
from time import time
from sklearn.linear_model import Lasso


#LASSO
print "doing a little lasso just to see what it gets to us :D..."
t0 = time()
regression = Lasso()
regression.fit(features, labels)
print "ok just fitted all of it 0_0, this took me about: ", round(time()-t0, 3), "s"

print "features and their coefs: "
for i in range(len(feature_all)):
    print "feature: ", feature_all[i]
    print " coef: ", regression.coef_[i]

#ENDLASSO

features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print'Previsao NB: '
print(pred)

### END Naive Bayes

### Random Forest

#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=50)

#clf.fit(features_train, labels_train)

#pred = clf.predict(features_test)
#print "Prediction RF: ", pred

# BEST Performance with all features
# acc: 0.847222222222
# precision: 0.4
# recall: 0.2

# BEST Performance with few features
# acc: 0.827586206897
# precision: 0.6
# recall: 0.272727272727
### End Decision Tree

### K neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

#clf = KNeighborsClassifier(weights='distance',n_neighbors=10)
#clf = BaggingClassifier(KNeighborsClassifier(weights='distance',n_neighbors=10))

#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#print'Previsao KN: '
#print(pred)

## K neighbours chuta tudo 0 =D
### END K neighbours

print "acc:"
print accuracy_score(labels_test,pred)

print "precision:"
print precision_score(labels_test,pred)

print "recall:"
print recall_score(labels_test, pred)

#print classification_report(labels_test, pred_kn, target_names=feature_all)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)