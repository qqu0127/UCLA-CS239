import scipy.io as sio
import numpy as np

dataFile = ['acc_data', 'adl_data', 'fall_data', 'two_classes_data']
labelFile = ['acc_labels', 'adl_labels', 'fall_labels', 'two_classes_labels']
nameFile = ['acc_names', 'adl_names', 'fall_names', 'two_classes_names']

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self) :
        """
        Data class.
        
        Attributes
        --------------------
            X -- numpy array of shape (n,d), features
            y -- numpy array of shape (n,), targets
        """
                
        # n = number of examples, d = dimensionality
        self.X = None
        self.y = None

        self.Xnames = None
        self.yname = None
        
    
    def fileInput(self, filename):
        path = "../data/" + filename + ".mat"
        temp_data = sio.loadmat(path)
        data = temp_data[filename]
        return data
    
    
    def extract_feature(self, data):
        self.Xnames = ["std_x", "std_y", "std_z", "max_sum_vector_magnitude"]
        n,d = data.shape
        sample_sum = int(n / 3)

        # feature use F2 & F5
        feature_sum = 4
        feature_matrix = np.zeros((sample_sum, feature_sum))
        mean_vec = np.mean(data, axis=1)
        std_vec = np.std(data, axis=1)

        i = 0
        while i != n:
            # F2 = the std of three axises
            feature_matrix[int(i / 3), 0] = std_vec[i]
            feature_matrix[int(i / 3), 1] = std_vec[i + 1]
            feature_matrix[int(i / 3), 2] = std_vec[i + 2]
            # F5 = the max Sum vector magnitude
            sum_vector_square = np.square(data[i]) + np.square(data[i + 1]) + np.square(data[i + 2])
            sum_vector_magnitude = np.sqrt(sum_vector_square)
            feature_matrix[int(i / 3), 3] = np.amax(sum_vector_magnitude)
            i += 3
        print (feature_matrix.shape)
        return feature_matrix

    def extract_label(self, label):
        self.yname = 'fall'
        n,d = label.shape
        label_sum = int(n / 3)

        label_vec = np.zeros(label_sum)
        i = 0
        while i != n:
            # in data, 1 means adl, 2 means fall 
            # in svm, -1 means adl, 1 means fall
            if (label[i, 0] == 1):
                label_vec[int(i / 3)] = -1
            else:
                label_vec[int(i / 3)] = 1
            i += 3
        return label_vec

    def load(self) :
        features = self.fileInput(dataFile[3])
        labels = self.fileInput(labelFile[3])

        self.X = self.extract_feature(features)
        self.y = self.extract_label(labels)
    


# helper functions
def load_data() :
    """Load .mat into Data class."""
    data = Data()
    data.load()
    return data
