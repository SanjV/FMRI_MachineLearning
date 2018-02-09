import util
import numpy


def test(firstCategory, secondCategory):
    def sigmoid(x, deriv=False):
        return x * (1 - x) if (deriv == True) else 1 / (1 + numpy.exp(-x))

    # initialization
    trainVoxelMap = util.getVoxelArray(allTasks=True, includeOddBall=False, includeWorkingMemory=False,includeSelectiveAttention=True,subjectNumbers=trainSubjects)
    testVoxelMap = util.getVoxelArray(allTasks=True, includeOddBall=False, includeWorkingMemory=False,includeSelectiveAttention=True,subjectNumbers=testSubjects)
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

    def testAcrossSubjects():
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

    numpy.random.seed(1)
    synapseZero = 2 * numpy.random.random((1973, 45)) - 1
    synapseOne = 2 * numpy.random.random((45, 1)) - 1

    for iter in range(5000):
        l0 = trainX
        l1 = sigmoid(numpy.dot(l0, synapseZero))
        l2 = sigmoid(numpy.dot(l1, synapseOne))

        l2_error = [trainY[i] - l2[i] for i in range(len(trainY))]
        l2_delta = l2_error * sigmoid(l2, deriv=True)

        l1_error = l2_delta.dot(synapseOne.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        synapseOne += l1.T.dot(l2_delta)
        synapseZero += l0.T.dot(l1_delta)


    synapse = numpy.dot(synapseZero, synapseOne)
    a = sigmoid(numpy.dot(testX, synapse))
    predictions = [0 if abs(a[i]) < abs(a[i] - 1)  else 1 for i in range(len(testX))]

    print ("Categories: ", category1, "/", category2, " Error: ", sum([1 if predictions[i] != testY[i] else 0 for i in range(len(testY))]), " out of ", len(testY), "accuracy", 100 - sum([1 if predictions[i] != testY[i] else 0 for i in range(len(testY))])/len(testY)*100, "%")
    #list  = 100 - (sum([1 if predictions[i] != testY[i] else 0 for i in range(len(testY))])/len(testY)*100)
    #print (list)

trainSubjects = [1, 2, 4]
testSubjects = [3]

# Default test to compare every pair of categories
for i in range(5):
    for j in range(i + 1, 5):
        if i != j:
            test(i, j)