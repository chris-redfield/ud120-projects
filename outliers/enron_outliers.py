#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

#removendo outlier sinistro que achamos
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)


salaries = []
bonuses = []
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    salaries.append(salary)
    bonuses.append(bonus)
    matplotlib.pyplot.scatter(salary,bonus)

#how to find the key of the biggest outlier (salary)
#keys = data_dict.keys()
#for poi in data_dict:
#    if data_dict[poi]['salary'] == max(salaries):
#        print poi
#        print data_dict[poi]

#print "max salary: " + str(max(salaries))
#print "max bonus: " + str(max(bonuses))

#print "ultra outlier object index: " + \
#      str(salaries.index(max(salaries))) + " - - -- - - -" + str(bonuses.index(max(bonuses)))

#finding the two most outlying points after that...
for poi in data_dict:

    if (data_dict[poi]['salary'] != "NaN") & (data_dict[poi]['bonus'] != "NaN"):

        if (data_dict[poi]['salary'] > 1000000) & (data_dict[poi]['bonus'] > 5000000):
            print "POI:" + poi
            print data_dict[poi]


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

