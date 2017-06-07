import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import csv

from util import *

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    m = metrics.confusion_matrix(y_true, y_label)
    if (metric == "accuracy"): return metrics.accuracy_score(y_true, y_label)
    if (metric == "f1_score"): return metrics.f1_score(y_true, y_label) 
    if (metric == "auroc"): return metrics.roc_auc_score(y_true, y_label)
    if (metric == "precision"): return metrics.precision_score(y_true, y_label)   
    if (metric == "sensitivity"): 
        TP = m[0, 0]
        FN = m[0, 1]
        if (TP + FN == 0.0):
            return 0.0
        else:
            return float(TP) / float(TP + FN)
    if (metric == "specificity"): 
        TN = m[1, 1]
        FP = m[0, 1]
        if (FP + TN == 0.0):
            return 0.0
        else:
            return float(TN) / float(FP + TN)
    return 0

def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    ave_performance = 0
    num = 0
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        ave_performance += performance(y_test, clf.decision_function(X_test), metric)
        num += 1
    return ave_performance / num

def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    for c in C_range:
        clf = SVC(kernel="linear", C=c)
        performance = cv_performance(clf, X, y, kf, metric)
        if (performance > p_max):
            p_max = performance
            c_return = c
        performance_list.append(performance)
    print performance_list
    return c_return

def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-4, 2)

    param_grid = []
    first_row = ['C/gamma']
    for gamma in gamma_range:
        first_row.append(gamma)
    param_grid.append(first_row)

    p_max = -1
    for c in C_range:
        row_c = []
        row_c.append(c)
        for gamma in gamma_range:
            clf = SVC(kernel="rbf", C=c, gamma=gamma)
            performance = cv_performance(clf, X, y, kf, metric)
            row_c.append(performance)
            if (performance > p_max):
                p_max = performance
                c_return = c
                g_return = gamma
            #performance_list.append(performance)
            print ("C = ", c, " Gamma=", gamma, " performance=", performance) 
        param_grid.append(row_c)

    """Write out predictions to csv file."""
    out = open("param_rbf_" + str(metric) + ".csv", 'wb')
    f = csv.writer(out)
    f.writerows(param_grid)
    out.close()

    return c_return, g_return

def main():
    '''
    data = load_data()
    X = data.X
    y = data.y
    '''
    temp_data = sio.loadmat('../demoFeaturesOutput/feat.mat')
    X = temp_data['feat']
    temp_label = sio.loadmat('../demoFeaturesOutput/labels.mat')
    y = temp_label['labels'][0]
    n,d = X.shape  # n = number of examples, d =  number of features
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    # for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    
    #skf = StratifiedKFold(y, n_folds=5)
    #for metric in metric_list:
    #    param_linear = select_param_linear(X, y, skf, metric)
    #    print("param_linear based on ", metric, "=", param_linear)


    # for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    #skf = StratifiedKFold(y, n_folds=5)
    #for metric in metric_list:
    #    param_rbf = select_param_rbf(X, y, skf, metric)
    #    print("param_rbf based on ", metric, "=", param_rbf)


    # split the train and test data
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    skf = StratifiedKFold(y_train, n_folds=5)
    clf_linear = SVC(kernel="linear", C=0.001)
    print '---------------------------------------'
    print "cv Performance for Linear svm"
    for metric in metric_list:
        ave_performance = cv_performance(clf_linear, X_train, y_train, skf, metric=metric)
        print(metric, "=", ave_performance)

    print '---------------------------------------'
    print "test Performance for Linear svm"
    clf_linear.fit(X_train, y_train)
    y_pred = clf_linear.decision_function(X_test)
    for metric in metric_list:
        test_performance = performance(y_test, y_pred, metric=metric)
        print(metric, "=", test_performance)

    print '---------------------------------------'
    print "cv Performance for RBF svm"
    clf_rbf = SVC(kernel="rbf", C=0.1, gamma=0.1)
    for metric in metric_list:
        ave_performance = cv_performance(clf_rbf, X_train, y_train, skf, metric=metric)
        print(metric, "=", ave_performance)

    print '---------------------------------------'
    print "test Performance for RBF svm"
    clf_rbf.fit(X_train, y_train)
    y_pred = clf_rbf.decision_function(X_test)
    for metric in metric_list:
        test_performance = performance(y_test, y_pred, metric=metric)
        print(metric, "=", test_performance)

if __name__ == "__main__" :
    main()
