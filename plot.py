import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

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

def main():
	data = Data()
	data.load()
	data.plot()

if __name__ == "__main__" :
    main()