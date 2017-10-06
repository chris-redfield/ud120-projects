#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    #remove 10% objects from the lists(biggest outliers) @_@
    outliers = int(len(predictions) * 0.1)

    cleaned_data = []

    print "outlier cleaner \n" \
          "Number of outliers: " + str(outliers)

    index = 0
    error = []
    #prepare set of errors
    while index < len(predictions):
        #ABS absolute value to evaluate biggest errors
        error.append(abs(net_worths[index][0] - predictions[index][0]))
        index += 1

    ages = ages.tolist()
    net_worths = net_worths.tolist()

    #removing the outlierssss
    while outliers > 0:
        index = error.index(max(error))
        del error[index]
        del ages[index]
        del net_worths[index]
        outliers -= 1

    # tudo que esta aqui poderia estar na primeira iteracao
    # mas eu fiquei com preguica -_-
    # sorry :~
    index = 0
    while index < len(error):
        tup = ages[index][0],net_worths[index][0],error[index]
        cleaned_data.append(tup)
        index += 1

    #thug life ;p
    ### your code goes here

    
    return cleaned_data

