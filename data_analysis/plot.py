import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz
import math

dataFile = ['acc_data', 'adl_data', 'fall_data', 'two_classes_data']
labelFile = ['acc_labels', 'adl_labels', 'fall_labels', 'two_classes_labels']
nameFile = ['acc_names', 'adl_names', 'fall_names', 'two_classes_names']

class Data :
    
    def __init__(self) :

        self.ave_magnitude = None  
        self.label_name = None   

    def fileInput(self, filename):
        path = "./data/" + filename + ".mat"
        temp_data = sio.loadmat(path)
        data = temp_data[filename]
        return data
        
    def cal_ave_magnitude(self, data, labels, names):
    	sample_sum, d = data.shape
    	self.label_name = []
    	label_type = names.shape[0]
    	
    	for name in names:
    		self.label_name.append(name[0][0])

    	# self.acc_magnitude = [[]]*(label - 1)
    	self.ave_magnitude = np.zeros((label_type, d))
    	acc_magnitude = [[] for k in range(label_type)]
        i = 0
        while i != sample_sum:
            # calculate the resultant acceleration a as the feature
            # a = sqrt(ax^2 + ay^2 + az^2)
            sum_vector_square = np.square(data[i]) + np.square(data[i + 1]) + np.square(data[i + 2])
            a = np.sqrt(sum_vector_square)
            #ci = self.extract_ci(a, n, m)
            label_i = labels[i, 0]
            acc_magnitude[label_i - 1].append(a)
            i += 3
        for k in range(label_type):
        	print(k+1, ' has ', len(acc_magnitude[k]))
        	acc_array = np.array(acc_magnitude[k])
        	print np.mean(acc_array, axis=0)
        	self.ave_magnitude[k, :] = np.mean(acc_array, axis=0)
        	
    def load(self) :
        data = self.fileInput(dataFile[0])
        labels = self.fileInput(labelFile[0])
        names = self.fileInput(nameFile[0])
        self.cal_ave_magnitude(data, labels, names)

    def plot(self):
		n,d = self.ave_magnitude.shape
		for i in range(n):
			plt.xlabel('Time')
			plt.ylabel('average magnitude')
			plt.plot(range(d), self.ave_magnitude[i, :])
			plt.title(self.label_name[i])
			plt.show()

    def low_pass_filter(self, a, label):
        #------------------------------------------------
        # Create a FIR filter and apply it to a.
        #------------------------------------------------
        d = float(len(a))
        sample_rate = 1.0
        nsamples = d
        t = np.arange(nsamples) / sample_rate
        # The Nyquist rate of the signal.
        nyq_rate = sample_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 5.0/nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 0.1

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        # Use lfilter to filter x with the FIR filter.
        filtered_a = lfilter(taps, 1.0, a)

        #ci = self.extract_ci(a, n, m)
        plt.xlabel('Time')
        plt.ylabel('average magnitude')
        plt.plot(t, a)

        delay = 0.5 * (N-1) / sample_rate
        plt.plot(t - delay, filtered_a, 'r-')
        plt.plot(t[N-1:] - delay, filtered_a[N-1:], 'g', linewidth=4)

        plt.title(label)
        plt.show()


    def plot_single(self, data, labels):
        sample_sum, d = data.shape
        i = 0
        while i != sample_sum:
            # calculate the resultant acceleration a as the feature
            # a = sqrt(ax^2 + ay^2 + az^2)
            sum_vector_square = np.square(data[i]) + np.square(data[i + 1]) + np.square(data[i + 2])
            a = np.sqrt(sum_vector_square)
            self.low_pass_filter(a, labels[i, 0])
            i += 3

    def load_single(self):
        data = self.fileInput(dataFile[2])
        labels = self.fileInput(labelFile[2])
        self.plot_single(data, labels)

def plot_histogram(X, y, Xname, show = True) :
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

def main():
    '''
    # plot raw data
	data = Data()
	data.load()
	data.plot()
    
    # plot single adl/fall
    data = Data()
    data.load_single()
    '''

    # plot feature
    temp_data = sio.loadmat('../demoFeaturesOutput/feat.mat')
    X = temp_data['feat']
    temp_label = sio.loadmat('../demoFeaturesOutput/labels.mat')
    y = temp_label['labels'][0]
    n,d = X.shape  # n = number of examples, d =  number of features

    #plot histograms of each feature
    print 'Plotting...'
    feature_name = ['std_x', 'std_y', 'std_z', 'max_magnitude', 'feature-5', 'feature-6', 'feature-7', 'feature-8']
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=feature_name[i])

if __name__ == "__main__" :
    main()