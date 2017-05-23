import scipy.io as sio
import numpy as np
import math
from sklearn import metrics
import csv
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

dataFile = ['acc_data', 'adl_data', 'fall_data', 'two_classes_data']
labelFile = ['acc_labels', 'adl_labels', 'fall_labels', 'two_classes_labels']
nameFile = ['acc_names', 'adl_names', 'fall_names', 'two_classes_names']

fall_size = 1699
adl_size = 5314

class Data :
    
    def __init__(self) :

        self.training_sample_set = None
        
        self.statistical_sample_set = None
        self.statistical_label_set = None

        self.test_sample_set = None
        self.test_label_set = None
        
    def extract_ci(self, feature, n, m):
        # 51 = 4 * 12 + 3 or 51 = 5 * 9 + 6
        # m = 4, n = 13
        b1 = 9
        b2 = 11
        distance = np.absolute(feature - b1) + np.absolute(feature - b2)
        ci = np.zeros(n)
        i = 0
        '''
        while i < 51:
            j = i
            i += m
            max_distance = -1
            arg_max = j
            while j != 51 and j < i:
                if (distance[j] >= max_distance):
                    max_distance = distance[j]
                    arg_max = j
                j += 1
            # aF can be symbolized by introducing a finite set of symbols: S = {1,2,3,4,5,6,7,8}
            af = feature[arg_max]
            if (af >= 0 and af < 3):
                ci[i / m - 1] = 0
            elif (af >= 3 and af < 6):
                ci[i / m - 1] = 1
            elif (af >= 6 and af < 9):
                ci[i / m - 1] = 2
            elif (af >= 9 and af <= 11):
                ci[i / m - 1] = 3
            elif (af > 11 and af < 19):
                ci[i / m - 1] = 4
            elif (af >= 19 and af < 27):
                ci[i / m - 1] = 5
            elif (af >= 27 and af < 35):
                ci[i / m - 1] = 6
            else:
                ci[i / m - 1] = 7
        '''
        while i < 51:
            if (feature[i] >= 0 and feature[i] < 3):
                ci[i] = 0
            elif (feature[i] >= 3 and feature[i] < 6):
                ci[i] = 1
            elif (feature[i] >= 6 and feature[i] < 9):
                ci[i] = 2
            elif (feature[i] >= 9 and feature[i] <= 11):
                ci[i] = 3
            elif (feature[i] > 11 and feature[i] < 19):
                ci[i] = 4
            elif (feature[i] >= 19 and feature[i] < 27):
                ci[i] = 5
            elif (feature[i] >= 27 and feature[i] < 35):
                ci[i] = 6
            else:
                ci[i] = 7
            i += 1
        # print ci
        return ci      

    def fileInput(self, filename):
        path = "../data/" + filename + ".mat"
        temp_data = sio.loadmat(path)
        data = temp_data[filename]
        return data
        
    def split_data(self, data, labels):
        # 51 = 4 * 12 + 3 or 51 = 5 * 9 + 6
        # m = 4, n = 13
        m = 4
        n = 51
        sample_sum = data.shape[0]
        fall_data = np.zeros((fall_size, n))
        adl_data = np.zeros((adl_size, n))
        fall_label = np.zeros(fall_size)
        adl_label = np.zeros(adl_size)
        i = 0
        index_adl = 0
        index_fall = 0
        while i != sample_sum:
            # calculate the resultant acceleration a as the feature
            # a = sqrt(ax^2 + ay^2 + az^2)
            sum_vector_square = np.square(data[i]) + np.square(data[i + 1]) + np.square(data[i + 2])
            a = np.sqrt(sum_vector_square)
            ci = self.extract_ci(a, n, m)
            #print ci
            # label(1~9) is adl, label(10~17) is fall
            if (labels[i, 0] < 10):
                adl_data[index_adl, :] = ci
                adl_label[index_adl] = labels[i, 0]
                index_adl += 1
                #print '--------adl----------'
                #print a
                #print ci
            else:
                fall_data[index_fall, :] = ci
                fall_label[index_fall] = labels[i, 0]
                index_fall += 1
                #print '--------fall----------'
                #print a
                #print ci
            i += 3

        print('adl number = ', index_adl, ' fall number = ', index_fall)

        split_fall = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in split_fall.split(fall_data, fall_label):
            fall_data_train, self.test_sample_set = fall_data[train_index], fall_data[test_index]
            fall_label_train, fall_label_test = fall_label[train_index], fall_label[test_index]
            self.test_label_set = np.ones(fall_label_test.shape[0])

            split_fall_train = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
            for training_index, statistical_index in split_fall_train.split(fall_data_train, fall_label_train):
                self.training_sample_set, self.statistical_sample_set = fall_data_train[training_index], fall_data_train[statistical_index]
                # non-use
                fall_label_training, fall_label_statistical = fall_label_train[training_index], fall_label_train[statistical_index]
                statistical_n, statistical_d = self.statistical_sample_set.shape
                #self.statistical_label_set = np.ones(statistical_n)
                self.statistical_label_set = fall_label_statistical
            print("fall training set number = ", self.training_sample_set.shape[0])
            print('fall statistical set number = ', self.statistical_sample_set.shape[0])
            print('fall test set number = ', self.test_sample_set.shape[0])

        split_adl = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for statistical_index, test_index in split_adl.split(adl_data, adl_label):
            adl_data_statistical = adl_data[statistical_index]
            adl_data_test = adl_data[test_index]

            print('adl statistical number = ', adl_data_statistical.shape[0], ' adl test number = ', adl_data_test.shape[0])

            # adl_label_statistical = np.zeros(adl_data_statistical.shape[0])
            # adl_label_test = np.zeros(adl_data_test.shape[0])
            # non-use
            adl_label_statistical = adl_label[statistical_index]
            adl_label_test = adl_label[test_index]
            self.statistical_sample_set = np.concatenate((self.statistical_sample_set, adl_data_statistical), axis=0)
            self.statistical_label_set = np.concatenate((self.statistical_label_set, adl_label_statistical), axis=0)
            self.test_sample_set = np.concatenate((self.test_sample_set, adl_data_test), axis=0)
            self.test_label_set = np.concatenate((self.test_label_set, adl_label_test), axis=0)
            print('statistical sample number = ', self.statistical_sample_set.shape[0])
            print('test sample number = ', self.test_sample_set.shape[0])
        
    def load(self) :
        data = self.fileInput(dataFile[0])
        labels = self.fileInput(labelFile[0])
        self.split_data(data, labels)

# helper functions
def load_data() :
    """Load .mat into Data class."""
    data = Data()
    data.load()
    return data

def main() :
    n = 51
    data = load_data()
    training_sample_set = data.training_sample_set
    statistical_sample_set = data.statistical_sample_set
    statistical_label_set = data.statistical_label_set
    test_sample_set = data.test_sample_set
    test_label_set = data.test_label_set

    print("fitting to HMM and decoding ...")
    # Make an HMM instance and execute fit
    training_sample_set_size = training_sample_set.shape[0]
    lengths = [n] * training_sample_set_size
    X = training_sample_set.reshape((n * training_sample_set_size, 1))
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X, lengths)
    # model = MultinomialHMM(n_components=3, n_iter=1000)
    model.fit(X, lengths)
    # Predict the optimal sequence of internal hidden state
    statistical_sample_sum = statistical_sample_set.shape[0]
    for i in range(statistical_sample_sum):
        obs = statistical_sample_set[i, :].transpose().reshape((n, 1))
        #print obs.shape
        logp = model.score(obs)
        print logp, statistical_label_set[i]

    print("done")
    
if __name__ == "__main__" :
    main()