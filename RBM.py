#RBM

import sys
import numpy as np
import math
from MNISTLoader import *
from HelperFunctions import *


class RBM(object):
	def __init__(self, visD, hidD, weightRange, groupWeights):
		self.visD = visD
		self.hidD = hidD

		self.initializeWeights(weightRange, groupWeights)

	def initializeWeights(self, valueRange, groupWeights):
		lowerbound = -valueRange/2
		upperbound = valueRange/2

		weightMatrix = np.random.uniform(lowerbound, upperbound, (self.hidD, self.visD))

		weightMatrixReg = np.random.randint(0, 2, (self.hidD, self.visD))
		weightMatrix = weightMatrix * weightMatrixReg
		weightMatrixReg = np.random.randint(0, 2, (self.hidD, self.visD))
		weightMatrix = weightMatrix * weightMatrixReg

		self.weightMatrix = weightMatrix
		self.weightMatrixShape = np.shape(weightMatrix)


	def generateHidStates(self, visStates):

		weightMatrix = self.weightMatrix.astype(np.float32)
		visStates = visStates.astype(np.float32)
		hidSignalMatrix = np.dot(weightMatrix, visStates)

		vectorizedLogistic = np.vectorize(logistic)

		result = vectorizedLogistic(hidSignalMatrix)
		return result


	def generateVisStates(self, hidStates):
		weightMatrixT = np.transpose(self.weightMatrix).astype(np.float32)
		hidStates = hidStates.astype(np.float32)
		visSignalMatrix = np.dot(weightMatrixT, hidStates)

		vectorizedLogistic = np.vectorize(logistic)

		result = vectorizedLogistic(visSignalMatrix)
		return result

	#Inputs should be actual visStates, and hidStates generated from those
	def calculateGrad(self, visStates, hidStates):

		samples = np.shape(visStates)[1]

		assert(samples == np.shape(hidStates)[1])

		hidStates = hidStates.astype(np.float32)
		visStatesT = np.transpose(visStates).astype(np.float32)
		weightTotals = np.dot(hidStates, visStatesT)

		result = weightTotals/samples

		resultShape = np.shape(result)

		assert(resultShape[0] == self.weightMatrixShape[0])
		assert(resultShape[1] == self.weightMatrixShape[1])

		return result


	#Take a matrix of probability values and tun into binary states
	def settleOnStates(self, probMatrix):

		randMatrix = np.random.uniform(0, 1, np.shape(probMatrix))

		diffMatrix = probMatrix - randMatrix

		resultMatrix = binaryConvertVectorized(diffMatrix)
		return resultMatrix

	#Same as settleOnStates but just goes to 0 or 1 with no randomness
	def goToStates(self, probMatrix):
		resultMatrix = binaryConvert2Vectorized(probMatrix)
		return resultMatrix


	def updateWeights(self, visStates, lr, cd):
		visStates = self.settleOnStates(visStates)		

		visStatesShape = np.shape(visStates)

		assert(visStatesShape[0] == self.visD)
		samples = visStatesShape[1]


		posHidStates = self.generateHidStates(visStates)
		posHidStates = self.settleOnStates(posHidStates)

		posHidStatesShape = np.shape(posHidStates)

		assert(posHidStatesShape[0] == self.hidD)
		assert(posHidStatesShape[1] == samples)

		posGrad = self.calculateGrad(visStates, posHidStates)

		currNegVisStates = visStates
		currNegHidStates = posHidStates


		for cdi in range(cd):
			currNegVisStates = self.generateVisStates(currNegHidStates)
			currNegVisStates = self.settleOnStates(currNegVisStates)

			currNegHidStates = self.generateHidStates(currNegVisStates)
			currNegHidStates = self.settleOnStates(currNegHidStates)


		negGrad = self.calculateGrad(currNegVisStates, currNegHidStates)

		finalGrad = posGrad - negGrad

		self.weightMatrix = self.weightMatrix + (lr*finalGrad) #I think you add it, maybe subtract though


	def train(self, inputMatrix, lr, iterations, batchSize):

		inputCases = np.shape(inputMatrix)[1]
		print("input cases:", inputCases)

		cd = 1

		for i in range(iterations):
			printProgress(i*100/iterations)		
			for batchi in range(int(inputCases/batchSize)):
				if (i > iterations/2):
					cd = 5

				batch = inputMatrix[:, batchSize*batchi:(batchSize*batchi)+batchSize]
				self.updateWeights(batch, lr, cd)

		print()


	def reconstructVisible(self, visState):
		visState = np.transpose(np.array([visState]))

		visState = self.settleOnStates(visState)

		hidState = self.generateHidStates(visState)
		hidState = self.settleOnStates(hidState)

		reconstructedVis = self.generateVisStates(hidState)

		printMNISTVector(reconstructedVis)
		
		return reconstructedVis


	def saveNet(self, fileName):
		f = open(fileName, 'wb')
		pickle.dump(self, f)
		f.close()
	

def loadNet(fileName):
	f = open(fileName, 'rb')
	network = pickle.load(f)
	f.close()
	return network



def logistic(z):
	try:
		return (1/(1+math.exp(-z)))
	except OverflowError:
		return 0


def binaryConvert(z):
	if (z < 0):
		return 0
	else:
		return 1

binaryConvertVectorized = np.vectorize(binaryConvert)

def binaryConvert2(z):
	if (z < .5):
		return 0
	else:
		return 1

binaryConvert2Vectorized = np.vectorize(binaryConvert2)

def setZeros(z):
	if ((z < .1) and (z > -.1)):
		return 0
	else:
		return z


def printProgress(percentComplete):
	percentString = "\rProgress: %d" % percentComplete
	percentString += "% "

	sys.stdout.flush()
	sys.stdout.write(percentString)



def main():


	load_last_network = sys.argv[1]


	iterations = 3
	
	hidUnits = 800
	lr = .5
	weightRange = 1.2
	cd = 1
	
	trainSubset = 2000
	validSusbet = 30
	testSubset = 0

	batchSize = 4

	training_data, validation_data, test_data = loadMINSTVectorSubset(trainSubset, validSusbet, testSubset)

	train_inputs = training_data[0]

	if (load_last_network == "s"):

		print("Training")
		print("Hidden Units:", hidUnits)
		print("LR:", lr, "WR:", weightRange, "CD:", cd)
		print("Iterations:", iterations, "batchsize", batchSize)
		print("Training Subset:", trainSubset)
	
		rbm = RBM(784, hidUnits, weightRange, True)

		rbm.train(train_inputs, lr, iterations, batchSize)
	

		rbm.saveNet("rbm")

	else:
		rbm = loadNet("rbm")

	testIndex = int(sys.argv[2])

	testInput = validation_data[0][:, testIndex]

	testLabelVector = validation_data[1][testIndex]

	printMNISTVector(testInput)

	rbm.reconstructVisible(testInput)




if __name__ == "__main__":
	main()
