#RBMStack

from RBM import *
import numpy as np
from AssertReport import *

class RBMStack(object):
	"""trainInfo component: wr, lr, cd, iterations, batchSize"""
	def __init__(self, dimensionList, trainInfoList):
		assert(len(dimensionList) == len(trainInfoList)+1)
		self.layers = len(trainInfoList)

		self.dimensionList = dimensionList
		self.trainInfoList = trainInfoList
		self.initRBMs()


	def getBotLayerDim(self):
		return self.dimensionList[0]

	def getTopLayerDim(self):
		return self.dimensionList[-1]


	def initRBMs(self):
		self.RBMList = []

		for i in range(self.layers):
			currVisDim = self.dimensionList[i]
			currHidDim = self.dimensionList[i+1]
			currWR = self.trainInfoList[i][0]
			groupWeights = False
			#if (i == 0):		#turning this off for cifar but use it for mnist
			#	groupWeights = True

			currRBM = RBM(currVisDim, currHidDim, currWR, groupWeights)
			self.RBMList.append(currRBM)




	def train(self, inputMatrix):

		currVisMatrix = inputMatrix

		for rbmi in range(self.layers):
			currRBM = self.RBMList[rbmi]
			currLR = self.trainInfoList[rbmi][1]
			currCD = self.trainInfoList[rbmi][2]
			currIterations = self.trainInfoList[rbmi][3]
			currBatchSize = self.trainInfoList[rbmi][4]

			print("Layer:", rbmi)
			#for i in range(currIterations):
				#print("Stack Iteration:", i)
				#if (i > currIterations/2):
				#	currCD = 5

				#currRBM.updateWeights(currVisMatrix, currLR, currCD)
				#currRBM.train(currVisMatrix, currLR, 1, currBatchSize)
			currRBM.train(currVisMatrix, currLR, currIterations, currBatchSize)

			currVisMatrix = currRBM.generateHidStates(currVisMatrix)




	def reconstructVis(self, visState):

		currVisState = visState

		for rbmi in range(self.layers):
			currRBM = self.RBMList[rbmi]
			assert(np.shape(currVisState)[0] == currRBM.visD)

			currVisState = currRBM.settleOnStates(currVisState)
			#currVisState = currRBM.goToStates(currVisState)
			currVisState = currRBM.generateHidStates(currVisState)


		for i in range(self.layers):
			rbmi = self.layers - 1 - i
			currRBM = self.RBMList[rbmi]

			assert(np.shape(currVisState)[0] == currRBM.hidD)

			currVisState = currRBM.settleOnStates(currVisState)
			#currVisState = currRBM.goToStates(currVisState)
			currVisState = currRBM.generateVisStates(currVisState)

		#printMNISTVector(currVisState)    NOTE THAT I TURNED THIS OFF
		return currVisState



	def generateNewVisState(self, iterations):

		topDim = self.dimensionList[-1]
		topRBM = self.RBMList[-1]

		currTopLayer = np.random.randint(0, 2, (topDim, 1))

		for i in range(iterations):

			currBotLayer = topRBM.generateVisStates(currTopLayer)

			currTopLayer = topRBM.generateHidStates(currBotLayer)

		#currVisState = topRBM.settleOnStates(currTopLayer)
		currVisState = currTopLayer
		for i in range(self.layers):
			rbmi = self.layers - 1 - i
			currRBM = self.RBMList[rbmi]

			#print(currRBM.weightMatrix)

			print("RBMI", rbmi)
			print("hidD", currRBM.hidD)
			print("visD", currRBM.visD)

			#print(currVisState)
			currVisState = currRBM.settleOnStates(currVisState)
			#print(currVisState)
			currVisState = currRBM.generateVisStates(currVisState)
			#print(currVisState)

		printMNISTVector(currVisState)
		return currVisState


	def feedForward(self, inputMatrix):
		currVisState = inputMatrix

		for rbmi in range(self.layers):
			currRBM = self.RBMList[rbmi]
			assert(np.shape(currVisState)[0] == currRBM.visD)

			currVisState = currRBM.settleOnStates(currVisState)
			#currVisState = currRBM.goToStates(currVisState)
			currVisState = currRBM.generateHidStates(currVisState)

		#result = currRBM.settleOnStates(currVisState)
		result = currVisState

		return result


	def saveNet(self, fileName):
		f = open(fileName, 'wb')
		pickle.dump(self, f)
		f.close()
	

def loadNet(fileName):
	f = open(fileName, 'rb')
	network = pickle.load(f)
	f.close()
	return network




def main():

	load_last_network = sys.argv[1]
	testIndex = int(sys.argv[2])
	#trainInfo component: wr, lr, cd, iterations

	
	#hidUnits1 = 700
	hidUnits1 = 1000
	lr = .5
	weightRange = 1.2
	cd = 1
	iterations = 10
	batchSize = 4

	trainInfo1 = [weightRange, lr, cd, iterations, batchSize]


	#hidUnits2 = 600
	hidUnits2 = 20
	lr = .5
	weightRange = 1.2
	cd = 1
	iterations = 10
	batchSize = 4

	trainInfo2 = [weightRange, lr, cd, iterations, batchSize]

	#hidUnits3 = 600
	#lr = .5
	#weightRange = 1
	#cd = 1
	#iterations = 40
	#batchSize = 2000
#
	#trainInfo3 = [weightRange, lr, cd, iterations, batchSize]

	
	trainSubset = 2000
	validSusbet = 30
	testSubset = 0

	training_data, validation_data, test_data = loadMINSTVectorSubset(trainSubset, validSusbet, testSubset)

	train_inputs = training_data[0]

	dimensionList = [784, hidUnits1, hidUnits2]#, hidUnits3]
	trainInfoList =[trainInfo1, trainInfo2]#, trainInfo3]
	
	rbmstack = RBMStack(dimensionList, trainInfoList)


	if (load_last_network == "s"):
		rbmstack.train(train_inputs)
		rbmstack.saveNet("rbmstack.txt")

	else:
		rbmstack = loadNet("rbmstack.txt")

	#testIndex = 3

	testInput = validation_data[0][:,testIndex]

	testLabelVector = validation_data[1][:,testIndex]
	#testLabel = outputVectorToLabel(testLabelVector)

	#printMNISTVector(testInput)
	#print("TestLabel:", testLabelVector)

	#rbmstack.reconstructVis(testInput)

	#rbmstack.generateNewVisState(0)
	#rbmstack.generateNewVisState(1)
	#rbmstack.generateNewVisState(15)
	rbmstack.generateNewVisState(500)


	#rbmstack.generateNewVisState(10)


if __name__ == "__main__":
	main()

