# UCLA-CS239
AI and Health Informatics -- course project

demo8Features.py :  --Reads in the features and labels extracted from Matlab script "featureExtractionDemo.m".
					--Conduct binary classification on ADL/FALL data using SVM.
					--Performance achieved so far:
							Performance for SVC
							accuracy = 0.937688434541
							f1_score = 0.865624074048
							auroc = 0.90062625129
							precision = 0.908641134572
							sensitivity = 0.9725261447808764
							specificity = 0.9086411345720414
					
MatlabUtl:			--Some useful Matlab helper function

featureExtractionDemo.m:
					--Read from raw data from folder "./data"
					--Process raw data and extract features for Python demo.
					--Output the features and labels to folder "./demoFeaturesOutput"