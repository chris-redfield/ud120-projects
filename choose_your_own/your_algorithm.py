#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

#cricks
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary


### Naive Bayes
#clf1 = GaussianNB()
#clf1.fit(features_train, labels_train)
#pred_nb = clf1.predict(features_test)

#print'Previsao NB: '
#print(pred_nb)

#acc_nb = accuracy_score(labels_test, pred_nb)
#print 'accuracy score NB: '
#print(acc_nb)
### END Naive Bayes

### K neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

clf2 = KNeighborsClassifier(weights='distance',n_neighbors=10)
#clf2 = BaggingClassifier(KNeighborsClassifier(weights='distance',n_neighbors=10))

clf2.fit(features_train, labels_train)
pred_kn = clf2.predict(features_test)

print'Previsao KN: '
print(pred_kn)

acc_kn = accuracy_score(labels_test, pred_kn)
print 'accuracy score KN: '
print(acc_kn)
### END K neighbours

### Ada Boost

#from sklearn.ensemble import AdaBoostClassifier

#clf3 = AdaBoostClassifier(n_estimators=100)

#clf3.fit(features_train, labels_train)

#pred_ab = clf3.predict(features_test)

#acc_ab = accuracy_score(labels_test, pred_ab)

#print 'accuracy score AB: '
#print(acc_ab)

### END Ada Boost


### Random Forest !!
#from sklearn.ensemble import RandomForestClassifier

#clf4 = RandomForestClassifier(n_estimators=50)

#clf4.fit(features_train, labels_train)

#pred_rf = clf4.predict(features_test)

#acc_rf = accuracy_score(labels_test, pred_rf)

#print 'accuracy score RF: '
#print(acc_rf)



### END Random Forest!!

try:
    prettyPicture(clf2, features_test, labels_test)
except NameError:
    pass
