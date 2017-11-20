#MNIST Loader

import pickle
import gzip
import numpy as np
from PIL import Image

def loadMNIST():
	f = gzip.open('MNIST_data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()

	#size = 50000
	trainingImages = training_data[0]
	trainingLabels = training_data[1]
	#trainingImages = training_data[0][0:10000]
	#trainingLabels = training_data[1][0:10000]

	#print(training_data[0][0])

	#print("training:", len(trainingImages))
	#print(trainingImages[0])
	#vectorizeLabel(trainingLabels[0])

	trainingCouples = coupleWithVLabels(trainingImages, trainingLabels)

	#validationImages = validation_data[0]
	#validationLabels = validation_data[1]
	validationImages = validation_data[0]
	validationLabels = validation_data[1]

	#print("validation", len(validationLabels))

	validationCouples = coupleWithVLabels(validationImages, validationLabels)

	#testImages = test_data[0]
	#testLabels = test_data[1]
	testImages = test_data[0][0:300]
	testLabels = test_data[1][0:300]

	#print("test", len(testLabels))


	testCouples = coupleWithVLabels(testImages, testLabels)


	#print(testCouples[0])

	return trainingCouples, validationCouples, testCouples


def loadMNISTVector():
	f = gzip.open('MNIST_data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()

	#training_data[0] = training_data[0].astype(np.float32)
	#validation_data[0] = validation_data[0].astype(np.float32)
	#test_data[0] = test_data[0].astype(np.float32)

	#training_data[1] = training_data[1].astype(np.float32)
	#validation_data[1] = validation_data[1].astype(np.float32)
	#test_data[1] = test_data[1].astype(np.float32)

	training_data = replaceLabelsWithVectors(training_data)
	validation_data = replaceLabelsWithVectors(validation_data)
	test_data = replaceLabelsWithVectors(test_data)



	#imageNum = 12
	#print(training_data[0][imageNum])
	#print(training_data[1][imageNum])
#
	#img = Image.new('L', (28, 28))
	#pixels = img.load()
#
	#imageVector = training_data[0][imageNum]
#
	#vectori = 0
	#for i in range(img.size[0]):
	#	for j in range(img.size[1]):
	#		pixels[j, i] = int(imageVector[vectori]*255)
	#		vectori += 1
#
	#img.show()

	#validation_data = replaceLabelsWithVectors(validation_data)
	#test_data = replaceLabelsWithVectors(test_data)

	#training_data = (addBias(training_data[0]), training_data[1])
	#validation_data = (addBias(validation_data[0]), validation_data[1])

	return training_data, validation_data, test_data




def loadMINSTVectorSubset(trainSubSize, validSubSize, testSubSize):
	trainingData, validationData, testData = loadMNISTVector()

	trainInput = trainingData[0]
	validInput = validationData[0]
	testInput = testData[0]


	trainInputSubset = trainInput[0:trainSubSize]
	validInputSubset = validInput[0:validSubSize]
	testInputSubset = testInput[0:testSubSize]


	trainLabels = trainingData[1]
	validLabels = validationData[1]
	testLabels = testData[1]

	trainLabelsSubset = trainLabels[0:trainSubSize]
	validLabelsSubset = validLabels[0:validSubSize]
	testLabelsSubset = testLabels[0:testSubSize]

	trainDataSubset = (np.transpose(trainInputSubset), np.transpose(trainLabelsSubset))
	validDataSubset = (np.transpose(validInputSubset), np.transpose(validLabelsSubset))
	testDataSubset =  (np.transpose(testInputSubset),  np.transpose(testLabelsSubset))

	#trainDataSubset = (trainInputSubset, trainLabelsSubset)
	#validDataSubset = (validInputSubset, validLabelsSubset)
	#testDataSubset =  (testInputSubset,  testLabelsSubset)

	return trainDataSubset, validDataSubset, testDataSubset




def vectorizeLabel(labelValue):
	vector = np.zeros((1, 10))[0]
	vector[labelValue] = 1
	#print("vector", vector)
	return vector

def coupleWithVLabels(imageVector, labelVector):
	coupleArray = []
	for i in range(len(imageVector)):
		#image = addBias(imageVector[i])
		vLabel = vectorizeLabel(labelVector[i])
		couple = [imageVector[i], vLabel]
		coupleArray.append(couple)
	return coupleArray


def replaceLabelsWithVectors(data):
	labelVector = data[1]
	newMatrix = np.zeros((len(labelVector), 10))	
	for i in range(len(labelVector)):
		label = labelVector[i]
		newMatrix[i] = vectorizeLabel(label)

	return (data[0], newMatrix)



def addBias(imageVector):
	oldVectorSize = len(imageVector)
	biasVector = np.zeros((1, oldVectorSize+1))[0]
	print(len(biasVector[0]))
	biasVector[0] = 1
	for i in range(oldVectorSize):
		biasVector[i+1] = imageVector[i]

	return biasVector


def main():
	loadMNISTVector()


if __name__ == "__main__":
	main()
