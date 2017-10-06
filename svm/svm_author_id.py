#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

#clf = svm.SVC(kernel='linear',C=10,gamma=10)

c = 10000

clf = svm.SVC(kernel='rbf',C=c)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred_svm = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print 'Previsao SVM:'
print(pred_svm)

chris_counter = 0

for pred in pred_svm:
    if pred == 1:
        chris_counter = chris_counter +1

print'Quantidade de vezes que adivinhei que seria o Chris: %d' % (chris_counter)

acc_svm = accuracy_score(labels_test,pred_svm)

print 'accuracy score SVM with C %d:' % (c)
print(acc_svm)


print 'pred[10]: %d' % (pred_svm[10])
print 'pred[26]: %d' % (pred_svm[26])
print 'pred[50]: %d' % (pred_svm[50])