#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit, createRelativeFeature
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectPercentile, f_classif

# overrides above var with features after selection
features_list = ['poi','bonus', 'exercised_stock_options', 'expenses', 'shared_receipt_with_poi','long_term_incentive','salary']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 3: Create new feature(s)
# Same algorithm as Lesson 12, 4. Vizualizing your new feature
createRelativeFeature(data_dict)

keys = data_dict.keys()

count_pois = 0
for key in keys:
    if data_dict[key]['poi'] == True:
        count_pois += 1

print "Total of people (DP): " + str(len(keys))
print "Total of POIs: " + str(count_pois)
print "Total of Non POIs: " + str(len(keys) - count_pois)
print "POI percentage: ", str(round(float(count_pois)/float(len(keys)) * 100,2)) +"%"

# quick analysis of the features
for feature in features_list:
    nan_counter = 0
    zero_counter = 0
    for key in keys:
        if data_dict[key][feature] == 'NaN':
            nan_counter += 1
        if data_dict[key][feature] == 0:
            zero_counter += 1
    print "# of NaNs for feature", feature, ": ", nan_counter
    print "# of zeros for feature", feature, ": ", zero_counter

#quick analysis of the data points, to see wich are irrelevant
for key in keys:
    nan_counter = 0
    for feature in features_list:
        if data_dict[key][feature] == 'NaN':
            nan_counter += 1
    #more like a rule of thumb, according to the number of features
    if nan_counter >= (len(features_list) - 1):
        print "deleting Datapoint", key, " for having only NaN values"
        #improve performance by removing datapoints with only NaN for this features
        data_dict.pop(key, 0)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
#already ripped off above, but just to keep documented
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# optimal found test_size = 70%
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.7, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=75, criterion='entropy',warm_start=True,oob_score=False)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print "Random Forest prediction + gridsearch tuned params: ", pred

print "acc:"
print accuracy_score(labels_test,pred)

print "precision:"
print precision_score(labels_test,pred)

print "recall:"
print recall_score(labels_test, pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
