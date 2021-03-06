import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

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
    print 'cv Performance of DecisionTreeClassifier'
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

def performance(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    m = metrics.confusion_matrix(y_test, y_pred)
    TP = m[0, 0]
    FN = m[0, 1]
    TN = m[1, 1]
    FP = m[0, 1]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    auroc = metrics.roc_auc_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    sensitivity = float(TP) / float(TP + FN)
    specificity = float(TN) / float(FP + TN)

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
    '''
    data = load_data()
    X = data.X; Xnames = data.Xnames
    y = data.y; yname = data.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    '''
    temp_data = sio.loadmat('../demoFeaturesOutput/feat.mat')
    X = temp_data['feat']
    temp_label = sio.loadmat('../demoFeaturesOutput/labels.mat')
    y = temp_label['labels'][0]
    n,d = X.shape  # n = number of examples, d =  number of features
    Xnames = ['std_x', 'std_y', 'std_z', 'max_magnitude', 'feature-5', 'feature-6', 'feature-7', 'feature-8']
    
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
    # split the train and test data
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = DecisionTreeClassifier(criterion = "entropy")
    print 'cv Performance for Decision Tree'
    cv_performance(clf, X_train, y_train)
    #d_train_error, d_test_error = error(clf, X, y)
    #print ('\t-- DecisionTreeClassifier: train_error: %.3f test_error: %.3f' % (d_train_error, d_test_error))

    print 'Investigating depths...'
    d_train_errors = []
    d_test_errors = []
    for i in range(1, 21):
        clf = DecisionTreeClassifier(criterion = "entropy", max_depth = i)
        d_train_error, d_test_error = error(clf, X_train, y_train)
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

    print 'test Performance for Decision Tree'
    clf = DecisionTreeClassifier(criterion = "entropy")
    performance(clf, X_train, y_train, X_test, y_test)




if __name__ == "__main__":
    main()