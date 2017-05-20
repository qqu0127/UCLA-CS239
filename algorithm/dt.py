import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        if (target == -1):
            labels.append('adl')
        else:
            labels.append('fall')
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels

def cv_performance(clf, X, y) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    print 'Performance of DecisionTreeClassifier'
    skf = StratifiedKFold(y, n_folds=5)
    # performance
    accuracy = 0.0
    f1_score = 0.0
    auroc = 0.0
    precision = 0.0
    sensitivity = 0.0
    specificity = 0.0

    ntrials = 0
    for train_index, test_index in skf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        ntrials += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # evaluate the performance
        m = metrics.confusion_matrix(y_test, y_pred)
        TP = m[0, 0]
        FN = m[0, 1]
        TN = m[1, 1]
        FP = m[0, 1]
        accuracy += metrics.accuracy_score(y_test, y_pred)
        f1_score += metrics.f1_score(y_test, y_pred)
        auroc += metrics.roc_auc_score(y_test, y_pred)
        precision += metrics.precision_score(y_test, y_pred)
        if (TP + FN == 0.0):
            sensitivity += 0.0
        else:
            sensitivity += float(TP) / float(TP + FN)
        if (FP + TN == 0.0):
            specificity += 0.0
        else:
            specificity += float(TN) / float(FP + TN)

    accuracy = accuracy / ntrials
    f1_score = f1_score / ntrials
    auroc = auroc / ntrials
    precision = precision / ntrials
    sensitivity = sensitivity / ntrials
    specificity = specificity / ntrials

    print("accuracy = ", accuracy)
    print("f1_score = ", f1_score)
    print("auroc = ", auroc)
    print("precision = ", precision)
    print("sensitivity = ", sensitivity)
    print("specificity = ", specificity)

def error(clf, X, y) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    skf = StratifiedKFold(y, n_folds=5)
    train_error = 0.0
    test_error = 0.0

    ntrials = 0
    for train_index, test_index in skf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        ntrials += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_train)
        train_error_single = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
        train_error += train_error_single

        y_pred = clf.predict(X_test)
        test_error_single = 1 - metrics.accuracy_score(y_test,y_pred,normalize=True)
        test_error += test_error_single

    train_error = train_error / ntrials
    test_error = test_error / ntrials
    return train_error, test_error

def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    data = load_data()
    X = data.X; Xnames = data.Xnames
    y = data.y; yname = data.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    #plot histograms of each feature
    '''
    print 'Plotting...'
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    '''
    
    
    '''
    # evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    clf = DecisionTreeClassifier(criterion = "entropy")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error

    # output the Decision Tree graph
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("dt_results/dtree.pdf") 
    '''

    # use cross-validation to compute average training and test error of classifiers
    clf = DecisionTreeClassifier(criterion = "entropy")
    cv_performance(clf, X, y)
    #d_train_error, d_test_error = error(clf, X, y)
    #print ('\t-- DecisionTreeClassifier: train_error: %.3f test_error: %.3f' % (d_train_error, d_test_error))

    print 'Investigating depths...'
    d_train_errors = []
    d_test_errors = []
    for i in range(1, 21):
        clf = DecisionTreeClassifier(criterion = "entropy", max_depth = i)
        d_train_error, d_test_error = error(clf, X, y)
        d_train_errors.append(d_train_error)
        d_test_errors.append(d_test_error)
    min = d_test_errors[0]

    # find minimum depth
    argmin = 1;
    for i in range(1, 21):
        if d_test_errors[i - 1] < min:
            min = d_test_errors[i - 1]
            argmin = i
    print argmin

    plt.plot(range(1, 21), d_train_errors, color='b', label='DecisionTree_train_error')
    plt.plot(range(1, 21), d_test_errors, color='r', label='DecisionTree_test_error')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=3, fancybox=True, shadow=True)
    plt.xlabel('depth limit')
    plt.ylabel('average error')
    plt.show()




if __name__ == "__main__":
    main()