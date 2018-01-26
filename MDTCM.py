import csv
import numpy as np
from plotDecBoundaries import plotDecBoundaries
import sys

# load csv file into numpy as matrices
a = np.loadtxt('wine_train.csv',dtype=float, delimiter=',')

# Define the sample numbers, feature numbers, class numbers
(sampleNum,featureNum)=a.shape
classNum = len(np.unique(a[:,13]))
classMean =np.zeros((classNum,featureNum))
label_train_done = np.zeros((sampleNum))
 
# Calculate class means for minimum-distance-to-class-means classifier, using Euclidean distance
for x in range(0,classNum):
	# for y in range(0,13):
	classMean[x,:]=np.mean((a[(a[:,13]==x+1),:]),axis=0)

# Train the classifier to find two features that have minimum error rate.
training_data=a[:,0:13]
minError=sys.maxsize
selectedFeature=(0,1)
for s in range(0,featureNum-3):
	for t in range(s+1, featureNum-2):
		for x in range(0,sampleNum):
			distance_Between_Class1_and_DataPoints= np.square(training_data[x,s]-classMean[0,s])+np.square(training_data[x,t]-classMean[0,t])
			distance_Between_Class2_and_DataPoints= np.square(training_data[x,s]-classMean[1,s])+np.square(training_data[x,t]-classMean[1,t])
			distance_Between_Class3_and_DataPoints= np.square(training_data[x,s]-classMean[2,s])+np.square(training_data[x,t]-classMean[2,t])
			if(min(distance_Between_Class1_and_DataPoints,distance_Between_Class2_and_DataPoints,distance_Between_Class3_and_DataPoints)==distance_Between_Class1_and_DataPoints):
				label_train_done[x]=1
			elif (min(distance_Between_Class1_and_DataPoints,distance_Between_Class2_and_DataPoints,distance_Between_Class3_and_DataPoints)==distance_Between_Class2_and_DataPoints):
				label_train_done[x]=2
			else:
				label_train_done[x]=3
		errorNum=(label_train_done != a[:,13]).sum()
		errorRate=errorNum/sampleNum
		if(minError!=min(errorRate,minError)):
			selectedFeature=(s+1,t+1)
		minError=min(errorRate,minError)

		# print(errorNum)
		# print(sampleNum)
		# print(errorNum/sampleNum)
print (selectedFeature)
print (minError)

plotDecBoundaries(a[:,0:13],a[:,13],classMean[:,0:2])

