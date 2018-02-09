import util
import numpy

def test(firstCategory, secondCategory):
	def sigmoid(x,deriv=False):
		return x*(1-x) if(deriv==True) else 1/(1+numpy.exp(-x))

	trainVoxelMap = util.getVoxelArray(allTasks=True, includeOddBall=False, includeWorkingMemory=False, includeSelectiveAttention=True,
	subjectNumbers = trainSubjects)
	testVoxelMap = util.getVoxelArray(allTasks=True, includeOddBall=False, includeWorkingMemory=False, includeSelectiveAttention=True,
	subjectNumbers = testSubjects)
	util.normalize(trainVoxelMap)
	util.normalize(testVoxelMap)

	trainX = []
	trainY = []
	testX = []
	testY = []

	category1 = firstCategory
	category2 = secondCategory

	def testWithinSubjects():
		for key in trainVoxelMap:
			if key[1] == category1 or key[1] == category2:
				classification = 1 if key[1] == category2 else 0
				if key[0] == 2:
					testX.append(trainVoxelMap[key])
					testY.append(classification)
				else:
					trainX.append(trainVoxelMap[key])
					trainY.append(classification)

	def testAcrossSubjects():	# include all data for a whole subject
		for key in trainVoxelMap:
			if key[1] == category1 or key[1] == category2:
				trainX.append(trainVoxelMap[key])
				classification = 1 if key[1] == category2 else 0
				trainY.append(classification)
		for key in testVoxelMap:
			if key[1] == category1 or key[1] == category2:
				testX.append(testVoxelMap[key])
				classification = 1 if key[1] == category2 else 0
				testY.append(classification)

	testWithinSubjects()

	trainX = numpy.array(trainX)
	trainY = numpy.array(trainY).T

	# back propogation
	numpy.random.seed(1)
	synapseZero = 2*numpy.random.random((1973, 1)) - 1
	for iter in range(10000):
		l0 = trainX
		l1 = sigmoid(numpy.dot(l0,synapseZero))
		l1_error = [ trainY[i]-l1[i] for i in range(len(trainY)) ]
		l1_delta = l1_error * sigmoid(l1, True)
		synapseZero += numpy.dot(l0.T, l1_delta)

	#print ("Correct Classifications: ", testY)
	a = sigmoid(numpy.dot(testX,synapseZero))
	predictions = [0 if abs(a[i]) < abs(a[i] - 1)  else 1 for i in range(len(testX))]
	#print "Output After Training: ", predictions
	print ("Categories: ", category1, "/", category2, " Error: ", sum([1 if predictions[i] != testY[i] else 0 for i in range(len(testY))]), " out of ", len(testY), "accuracy", 100 - sum([1 if predictions[i] != testY[i] else 0 for i in range(len(testY))])/len(testY)*100, "%")


trainSubjects = [1,2,4]
testSubjects = [3]

# Default test to compare every pair of categories
for i in range(5):
	for j in range(i+1,5):
		if i != j:
			test(i, j)