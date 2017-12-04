#MNIST Loader

import pickle
import gzip
import numpy as np


def loadMNISTVector(mnistFile):
	try:
		f = gzip.open(mnistFile, 'rb')
		training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
		f.close()
	except:
		print("ERROR\nFailed to open MNIST data. Please be sure the proper directory "+
			"is specified and the file is in the format of <filename>.pkl.gz")
		exit()

	return training_data, validation_data, test_data


def loadMINSTVectorSubset(mnistFile, trainSubSize, validSubSize, testSubSize):
	trainingData, validationData, testData = loadMNISTVector(mnistFile)

	trainInputSubset = trainingData[0][:trainSubSize,:]
	validInputSubset = validationData[0][:validSubSize,:]
	testInputSubset = testData[0][:testSubSize,:]


	trainLabelsSubset = replaceLabelsWithVectors(trainingData[1][:trainSubSize])
	validLabelsSubset = replaceLabelsWithVectors(validationData[1][:validSubSize])
	testLabelsSubset = replaceLabelsWithVectors(testData[1][:testSubSize])


	trainDataSubset = (np.transpose(trainInputSubset), np.transpose(trainLabelsSubset))
	validDataSubset = (np.transpose(validInputSubset), np.transpose(validLabelsSubset))
	testDataSubset =  (np.transpose(testInputSubset),  np.transpose(testLabelsSubset))


	return trainDataSubset, validDataSubset, testDataSubset



def vectorizeLabel(labelValue):
	vector = np.zeros((1, 10))[0]
	vector[labelValue] = 1
	return vector


def replaceLabelsWithVectors(labelVector):
	sevens = 0

	vectors = np.shape(labelVector)[0]
	newMatrix = np.zeros((vectors, 10))	
	for i in range(vectors):
		label = labelVector[i]
		newMatrix[i, label] = 1

	return newMatrix


def main():
	loadMNISTVector()


if __name__ == "__main__":
	main()
