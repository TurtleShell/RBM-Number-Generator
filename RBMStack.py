#RBMStack

from RBM import *
import numpy as np

class RBMStack(object):
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
			currRBM.train(currVisMatrix, currLR, currIterations, currBatchSize)

			currVisMatrix = currRBM.generateHidStates(currVisMatrix)



	def reconstructVis(self, visState):

		currVisState = visState

		for rbmi in range(self.layers):
			currRBM = self.RBMList[rbmi]
			assert(np.shape(currVisState)[0] == currRBM.visD)

			currVisState = currRBM.settleOnStates(currVisState)
			currVisState = currRBM.generateHidStates(currVisState)


		for i in range(self.layers):
			rbmi = self.layers - 1 - i
			currRBM = self.RBMList[rbmi]

			assert(np.shape(currVisState)[0] == currRBM.hidD)

			currVisState = currRBM.settleOnStates(currVisState)
			currVisState = currRBM.generateVisStates(currVisState)

		return currVisState



	def generateNewVisState(self, iterations):

		topDim = self.dimensionList[-1]
		topRBM = self.RBMList[-1]

		currTopLayer = np.random.randint(0, 2, (topDim, 1))

		for i in range(iterations):
			currBotLayer = topRBM.generateVisStates(currTopLayer)
			currTopLayer = topRBM.generateHidStates(currBotLayer)

		currVisState = currTopLayer
		for i in range(self.layers):
			rbmi = self.layers - 1 - i
			currRBM = self.RBMList[rbmi]

			currVisState = currRBM.settleOnStates(currVisState)
			currVisState = currRBM.generateVisStates(currVisState)

		printMNISTVector(currVisState)
		return currVisState


	def feedForward(self, inputMatrix):
		currVisState = inputMatrix

		for rbmi in range(self.layers):
			currRBM = self.RBMList[rbmi]
			assert(np.shape(currVisState)[0] == currRBM.visD)

			currVisState = currRBM.settleOnStates(currVisState)
			currVisState = currRBM.generateHidStates(currVisState)

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

	mnistFile = sys.argv[2]


	hidUnits1 = 1000
	lr = 1
	weightRange = 1.2
	cd = 1
	iterations = 50
	batchSize = 4

	trainInfo1 = [weightRange, lr, cd, iterations, batchSize]


	hidUnits2 = 20
	lr = 1
	weightRange = 1.2
	cd = 1
	iterations = 50
	batchSize = 4

	trainInfo2 = [weightRange, lr, cd, iterations, batchSize]



	
	trainSubset = 2000
	validSusbet = 0
	testSubset = 0

	training_data, validation_data, test_data = loadMINSTVectorSubset(mnistFile, trainSubset, validSusbet, testSubset)

	train_inputs = training_data[0]

	dimensionList = [784, hidUnits1, hidUnits2]
	trainInfoList =[trainInfo1, trainInfo2]
	
	rbmstack = RBMStack(dimensionList, trainInfoList)


	if (load_last_network == "s"):
		rbmstack.train(train_inputs)
		rbmstack.saveNet("rbmstack")

	else:
		rbmstack = loadNet("rbmstack")


	rbmstack.generateNewVisState(500)



if __name__ == "__main__":
	main()

