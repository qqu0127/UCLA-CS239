* SVM
  1. one Linear svm, one RBF svm
  2. Line 189 - 199 are used for choosing parameters for svm
  3. performance results are in 'svm_results/performance.txt'
  4. 'svm_results/param_linear.txt' shows parameter's influence on performances of linear svm.
  5. 'svm_results/param_rbf_*.csv' shows parameter's influence on performances of rbf svm.

* DecisionTree
  1. 'dt_results/dt_depth.png' shows relationship between dt's depth and error
  2. 'dt_results/dtree.pdf' shows the decision tree we construct
  3. 'dt_results/performance.txt' shows the performance of dt

* Feature-Extraction(util.py)
  1. now only use F2(Standarddeviation) and F5(Sum vector magnitude). F2 corresponds to three features on three axes. So totally four features. As for F5, here we use the highest peak during the sudden change of this value as the feature.
  2. 'feature_distribution' shows the feature value distribution of adl/fall
