#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import sys
import pickle
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

keys = enron_data.keys()
#features = enron_data['DONAHUE JR JEFFREY M'].keys()
features = ["poi", "total_payments","salary"]
np = featureFormat(enron_data, features,remove_NaN=True)

print np

#print enron_data["SKILLING JEFFREY K"]["bonus"]
counter = 0

for poi in enron_data:
    if enron_data[poi]['poi'] == True:
        print enron_data[poi]['poi']
        print enron_data[poi]['total_payments']
        counter += 1
print counter

#for poi in enron_data:
#    if enron_data[poi]['total_payments'] == 'NaN':
#        print enron_data[poi]['total_payments']
#        print enron_data[poi]['poi']
#        counter +=1

#print counter

##146 = 100%
#21 = 14.38#

#POI = chave
#for poi in enron_data:
    #print poi

    #mostra cada objetos full
    #print enron_data[poi]

    #mostra a feature 'poi' de cada objeto
    #print enron_data[poi]['poi']

    # conta se e POI
    #if enron_data[poi]["poi"] == True:
    #    counter += 1

    #print poi + ": " + str(enron_data[poi]['total_stock_value'])
    #print poi + " salary: " + str(enron_data[poi]['salary'])
    #print poi + " email: " + str(enron_data[poi]['email_address'])

    #if enron_data[poi]['email_address'] != 'NaN':
     #   print poi + " salary: " + str(enron_data[poi]['salary'])
     #   counter += 1
#print counter

#print "James Prentice:"
#print enron_data['PRENTICE JAMES']['total_stock_value']

#print "LAY: " + str(enron_data['LAY KENNETH L']['total_payments'])
#print "SKILLING: " + str(enron_data['SKILLING JEFFREY K']['total_payments'])
#print "FASTOW: " + str(enron_data['FASTOW ANDREW S']['total_payments'])
#print counter