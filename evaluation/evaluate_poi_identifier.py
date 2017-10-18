#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print "labels test"
print labels_test

print "pred test"
print pred

print "true - pred"
for i in range(len(pred)):
    print str(labels_test[i]) + " - " + str(pred[i])

#for n,i in enumerate(pred):
#    if i==1:
#        pred[n]=0
#    else:
#        pred[n]=1

#print "reverse pred"
#print pred

#for i in range(len(pred)):
#    pred[i] = 0.

#print "all 0 pred"
#print pred

print "acc:"
print accuracy_score(labels_test,pred)

print "precision:"
print precision_score(labels_test,pred)

print "recall:"
print recall_score(labels_test, pred)