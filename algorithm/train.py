import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn import tree
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import svm 
from sklearn import neighbors
import scipy.io as sio
import matplotlib.pyplot as plt

def performance(y_true, y_pred):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions       
    
    Returns
    --------------------
        metric -- ndarray, performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity' 
    """   

    m = metrics.confusion_matrix(y_true, y_pred)
    #print m
    metric = np.zeros(6)
    metric[0] = metrics.accuracy_score(y_true, y_pred)
    metric[1] = metrics.f1_score(y_true, y_pred) 
    metric[2] = metrics.roc_auc_score(y_true, y_pred)
    metric[3] = metrics.precision_score(y_true, y_pred)   
    
    TP = m[0, 0]
    FN = m[0, 1]
    metric[4] = float(TP) / float(TP + FN)

    TN = m[1, 1]
    FP = m[0, 1]
    metric[5] = float(TN) / float(FP + TN)
    
    return metric

def cv_performance(clf, X, y, kf):
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
    
    Returns
    --------------------
        metric -- ndarray, performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity' 
    """
    metric = np.zeros((5, 6))
    index = 0
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        metric[index, :] = performance(y_test, clf.predict(X_test))
        index += 1
        # metric = metric + performance(y_test, clf.predict(X_test))
        # num += 1
    # metric = metric / num
    # print metric
    return np.mean(metric, axis=0), np.std(metric, axis=0)


temp_data = sio.loadmat('../demoFeaturesOutput/feat.mat')
X = temp_data['feat']
temp_label = sio.loadmat('../demoFeaturesOutput/labels.mat')
y = temp_label['labels'][0]
n,d = X.shape  # n = number of examples, d =  number of features

## split the train and test data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

skf = StratifiedKFold(y_train, n_folds=5)

#cls = 'tree' 
#cls = 'bayes' 
#cls = 'GBDT' 
#cls = 'lr' 
#cls = 'svm' 
#cls = 'rf' 
#cls = 'knn' 
model = None
model_name = 'tree'

df_metric = pd.DataFrame(columns=("model", "accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"))
metric = np.zeros((7, 6))

cls = clf = ensemble.RandomForestClassifier(max_depth = 5)
clf.fit(X, y)
# print clf.feature_importances_
features = ['std_x', 'std_y','std_z', 'average_magnitude', 'feature5', 
            'feature6', 'feature7', 'feature8']
fi = clf.feature_importances_
opacity = 0.2
plt.bar(range(len(features)), fi, alpha=opacity)

plt.xticks(range(len(features)), features)
plt.xticks(rotation = 45)
plt.show()

score = 0
index = 0
for cls in ['tree', 'bayes', 'GBDT', 'svm', 'lr', 'rf', 'knn']:
    if cls == 'tree':
      clf = tree.DecisionTreeClassifier()
    if cls == 'bayes':
      clf = naive_bayes.GaussianNB()
    if cls == 'GBDT':
      clf = ensemble.GradientBoostingClassifier()
    if cls == 'lr':
      clf = linear_model.LogisticRegression()
    if cls == 'svm':  
      clf = svm.SVC(kernel = 'linear')
    if cls == 'rf':
      clf = ensemble.RandomForestClassifier(max_depth = 5)
    if cls == 'knn':  
      clf = neighbors.KNeighborsClassifier()
    
    print "cv Performance for " + cls
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    average, std = cv_performance(clf, X_train, y_train, skf)
    metric[index, :] = average
    index += 1

    print metric_list
    print average
    print std

    '''
    df_metric.loc[index] = {'model': cls, 
                            'accuracy': str(average[0]) + '+-' + str(std[0]),
                            'f1_score': str(average[1]) + '+-' + str(std[1]),
                            'auroc': str(average[2]) + '+-' + str(std[2]),
                            'precision': str(average[3]) + '+-' + str(std[3]),
                            'sensitivity': str(average[4]) + '+-' + str(std[4]),
                            'specificity': str(average[5]) + '+-' + str(std[5])}
    index += 1
    '''
    if average[0] > score:
        score = average[0]
        model = clf
        model_name = cls
# print metric
# df_metric.to_csv("train_results.csv")

accuracy_list = metric[:, 0]
#print accuracy_list 
sensitivity_list = metric[:, 4]
specificity_list = metric[:, 5]


model_range = np.array(range(1, 8))
plt.figure(figsize=(14,10))
opacity = 0.2
bar_width = 0.2
plt.bar(model_range - 0.1, accuracy_list, 
        bar_width, alpha=opacity, color='c', label='acuracy')
plt.bar(model_range - 0.3, sensitivity_list, 
            bar_width, alpha=opacity, color='g', label='sensitivity')
plt.bar(model_range + 0.1, specificity_list, 
            bar_width, alpha=opacity, color='m', label='specificity')

plt.xlabel('model')
plt.ylabel('score')
plt.xticks(model_range, ('tree', 'bayes', 'GBDT', 'svm', 'lr', 'rf', 'knn'), rotation=45)
plt.title('Model Test Performance')
plt.tight_layout() 
plt.legend(prop={'size':9})
plt.show()

print model_name # GBDT
clf = model

# test the model with test-set
print 'test Performance for ' + model_name
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metric = performance(y_test, y_pred)
print metric_list
print metric

