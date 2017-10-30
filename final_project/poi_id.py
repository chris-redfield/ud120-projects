#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit, createRelativeFeature
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectPercentile, f_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# all features except email_address
# added the 2 new features, fraction_from_poi and fraction_to_poi
# still keeping loan_advances and restricted_stock_deferred
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock',
                 'shared_receipt_with_poi', 'restricted_stock_deferred',
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
                 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person', 'fraction_from_poi', 'fraction_to_poi']

# overrides above var with features after selection
features_list = ['poi','bonus', 'exercised_stock_options', 'expenses', 'shared_receipt_with_poi','long_term_incentive','salary']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Creating new feature
# Same algorithm as Lesson 12, 4. Vizualizing your new feature
createRelativeFeature(data_dict)

keys = data_dict.keys()

count_pois = 0
for key in keys:
    #print data_dict[key]

    if data_dict[key]['poi'] == True:
        print "poi:",key
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


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##remove poi label for lasso
#del features_list[features_list.index('poi')]

#LASSO regression, uncomment above for this to work, because of 'poi' feature
#from sklearn.linear_model import Lasso

#print "doing a lasso to see the best features"
#regression = Lasso(positive=True)
#regression.fit(features, labels)

#print "features and their coefs: "
#for i in range(len(features_list)):
#    print "feature: ", sorted(features_list)[i]
#    print " coef: ", regression.coef_[i]

#ENDLASSO

### remove low variance
#from sklearn.feature_selection import VarianceThreshold

#selector = VarianceThreshold()
#features = selector.fit_transform(features)
#ya didnt work so well ...
### end remove low variance

#the test size was so important omg...
# optimal found test_size = 70%
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(features, labels, test_size=0.7, random_state=42)

### Select Percentile
#selector = SelectPercentile(f_classif,percentile=50)
#selector.fit(features_train,labels_train)
#features_train_transformed = selector.transform(features_train)
#features_test_transformed = selector.transform(features_test)

### END Select Percentile

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Feature scaling

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# try to scale directly
features_train_scaled = preprocessing.scale(features_train)

#use minmax to see if there is a relevant difference
min_max_scaler = preprocessing.MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.fit_transform(features_test)
#well that didnt go so well, gonna rollback this =D
### end Feature scaling



# Provided to give you a starting point. Try a variety of classifiers.

### Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#print'Previsao NB: '
#print(pred)

# performance
# acc: 0.84375
# precision: 0.333333333333
# recall: 0.363636363636

# not possible to use gridsearch because it only has priors parameter

### END Naive Bayes

### Decision Tree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier(min_samples_split=20)
#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#print'Previsao DT: '
#print(pred)

# Performance
# acc: 0.84375
# precision: 0.25
# recall: 0.181818181818

# gridsearch

from sklearn import tree
#from sklearn import grid_search

#parameters = {'criterion':['entropy','gini'],'splitter':['random','best'],
#              'min_samples_split':[2,5,10,15,20,40,60], }
#dt = tree.DecisionTreeClassifier()
#clf = grid_search.GridSearchCV(dt, parameters)
#clf.fit(features_train, labels_train)
#print clf.best_params_
# prints {'min_samples_split': 10, 'splitter': 'random', 'criterion': 'entropy'}
# prints {'min_samples_split': 5, 'splitter': 'random', 'criterion': 'entropy'}

#clf = tree.DecisionTreeClassifier(min_samples_split=10, splitter='random', criterion='entropy')
#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#print'Previsao DT + gridsearch: '
#print(pred)

# Performance
# acc: 0.885416666667
# precision:0.5
# recall: 0.181818181818

# gridsearch

### END Decision Tree


### Random Forest

from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=50)

#clf.fit(features_train, labels_train)

#pred = clf.predict(features_test)
#print "Prediction RF: ", pred

#gridsearch

from sklearn import grid_search
#parameters = {'n_estimators':[25, 50, 75, 100], 'criterion':['entropy', 'gini'],
 #             'warm_start':[True,False], 'oob_score':[True,False]}
#rf = RandomForestClassifier()
#clf = grid_search.GridSearchCV(rf, parameters)
#clf.fit(features_train, labels_train)
#print clf.best_params_

#prints: {'n_estimators': 25, 'warm_start': False, 'criterion': 'entropy'}
#prints: {'n_estimators': 75, 'warm_start': True, 'oob_score': False, 'criterion': 'entropy'}


clf = RandomForestClassifier(n_estimators=75, criterion='entropy',warm_start=True,oob_score=False)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

# test just to see scaling being ineffective in trees :D
#clf.fit(features_train_scaled, labels_train)
#pred = clf.predict(features_test_scaled)

print "Prediction RF + gridsearch tuned params: ", pred


#gridsearch


# best performance with all features
# acc: 0.861386138614
# precision: 0.5
# recall: 0.142857142857


# best performance with few features (handpicked after lasso + gridsearch)
# acc: 0.916666666667
# precision: 0.8
# recall: 0.363636363636

# best performance with few features (Select Percentile)
# acc: 0.833333333333
# precision: 0.25
# recall: 0.1

### End Random Forest

### K neighbours
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import BaggingClassifier

#clf = KNeighborsClassifier(weights='distance',n_neighbors=10)
#clf = BaggingClassifier(KNeighborsClassifier(weights='distance',n_neighbors=10))

#clf.fit(features_train_scaled, labels_train)
#pred = clf.predict(features_test_scaled)

#print'Previsao KN: '
#print(pred)

## K neighbours chuta tudo 0 =D
### END K neighbours

### SVM

#from sklearn.svm import SVC
#clf = SVC()

#clf.fit(features_train_scaled, labels_train)
#clf.fit(features_train, labels_train)


#pred = clf.predict(features_test_scaled)
#pred = clf.predict(features_test)


#print'Previsao KN: '
#print(pred)

#SVM chutou tudo 0, e nao mudou acc apos reducao de variaveis

### END SVM


### adaboost

#from sklearn.ensemble import AdaBoostClassifier

#clf = AdaBoostClassifier(n_estimators=100)
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#print "Adabost preds"
#print pred

# gridsearch

from sklearn import grid_search
#parameters = {'n_estimators':[25,50,75,100], 'learning_rate':[1,2,3], 'algorithm': ['SAMME', 'SAMME.R']}
#ab = AdaBoostClassifier()
#clf = grid_search.GridSearchCV(ab, parameters)
#clf.fit(features_train, labels_train)
#print clf.best_params_
#prints {'n_estimators': 25, 'learning_rate': 2, 'algorithm': 'SAMME'}

#clf = AdaBoostClassifier(n_estimators=25, learning_rate=2, algorithm='SAMME')
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#print "Adabost preds + gridsearch"
#print pred

#lol precision has lowered after the gridsearch @_@

# gridsearch

# Resultado com todas as features
#acc: 0.861386138614
#precision: 0.5
#recall: 0.357142857143

#acc: 0.886363636364
#precision: 0.5
#recall: 0.2

#acc: 0.861386138614
#precision: 0.5
#recall: 0.285714285714

# Resultado com features do christian
# acc: 0.80487804878
# precision: 0.461538461538
# recall: 0.4

# Resultado com 25% das melhores features (select percentile)
#acc: 0.75
#precision: 0.1
#recall: 0.1

### END ab

print "true label - pred"
for index in range(len(pred)):
    print labels_test[index],"   -  ",pred[index]

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