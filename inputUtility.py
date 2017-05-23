# Author: Qi Qu

# Matlab - Python interface
# Wrap the data into ndarray to allow access from Python

# dataInput():
#INPUT: one of the dataNames
#OUTPUT: Python ndarray

import scipy.io as sio

dataNames = ['acc_data', 'adl_data', 'fall_data', 'two_classes_data']
labelNames = ['acc_labels', 'adl_labels', 'fall_labels', 'two_classes_labels']
nameFile = ['acc_names', 'adl_names', 'fall_names', 'two_classes_names']


def dataInput(filename):
    path = "./data/" + filename + ".mat"
    temp_data = sio.loadmat(path)
    data = temp_data[filename]
    return data
	
	
def dataInputAll():
	d1 = dataInput(dataNames[0])
	d2 = dataInput(dataNames[1])
	d3 = dataInput(dataNames[2])
	d4 = dataInput(dataNames[3])
	return d1, d2, d3, d4

def main() :
	d = dataInput(labelNames[0])
	r,c = d.shape
	print d
	
if __name__ == "__main__" :
    main()