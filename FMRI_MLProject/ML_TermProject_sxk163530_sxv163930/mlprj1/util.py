import scipy.io
import random
import math
import Constants
import numpy


def getDistance(v1, v2):
    # centroid/value are each lists of length 1973 (one for each voxel)
    distancesByVoxel = [math.pow(v1[i] - v2[i], 2) for i in range(len(v1))]
    distancesSquared = sum(distancesByVoxel)
    return distancesSquared


def dataStats(voxelArrayMap):
    for key in voxelArrayMap:
        print(min(voxelArrayMap[key]), max(voxelArrayMap[key]), key)


def normalize(voxelArrayMap):
    numFeatures = len(voxelArrayMap[list(voxelArrayMap.keys())[0]])
    m = len(voxelArrayMap)
    average = [0] * numFeatures

    # calculate average
    for key in voxelArrayMap:
        x = voxelArrayMap[key]
        for index in range(numFeatures):
            average[index] += x[index] / m

    # replace x's
    for key in voxelArrayMap:
        x = voxelArrayMap[key]
        for index in range(numFeatures):
            x[index] -= average[index]

    # calculate variance
    variance = [0] * numFeatures
    for key in voxelArrayMap:
        for i in range(numFeatures):
            variance[i] += voxelArrayMap[key][i] * voxelArrayMap[key][i]

    # update x with x / std
    for key in voxelArrayMap:
        for i in range(numFeatures):
            if variance[i] == 0:
                variance[i] = 1
            voxelArrayMap[key][i] /= math.sqrt(variance[i])


# return three dataSets out of this
def parse(file, subjectKey):
    mat = scipy.io.loadmat(file + '.mat')
    sub_Betas = mat[file]

    oddBall = numpy.zeros((Constants.NUM_VOXELS, Constants.NUM_CATEGORIES, Constants.NUM_RUNS))
    workingMemory = numpy.zeros((Constants.NUM_VOXELS, Constants.NUM_CATEGORIES, Constants.NUM_RUNS))
    selectiveAttention = numpy.zeros((Constants.NUM_VOXELS, Constants.NUM_CATEGORIES, Constants.NUM_RUNS))
    for v in range(Constants.NUM_VOXELS):
        for cat in range(15):
            for run in range(Constants.NUM_RUNS):
                imageCat = cat / 3
                if cat % 3 == 0:
                    oddBall[int(v)][int(imageCat)][int(run)] = sub_Betas[int(v)][int(cat)][int(run)]
                elif cat % 3 == 1:
                    workingMemory[int(v)][int(imageCat)][int(run)] = sub_Betas[int(v)][int(cat)][int(run)]
                else:
                    selectiveAttention[int(v)][int(imageCat)][int(run)] = sub_Betas[int(v)][int(cat)][int(run)]
    return [(oddBall, 'oddBall', subjectKey), (workingMemory, 'workingMemory', subjectKey),
            (selectiveAttention, 'selectiveAttention', subjectKey)]


def containsOne(subjectNumbers):
    for num in subjectNumbers:
        if num == 1:
            return True
    return False


def filterData(data, categories=[0, 1, 2, 3, 4]):
    for key in list(data):
        run, category, taskName, subject = key
        if category not in categories:
            del data[key]
    return data


def dimensionReduction(data):
    factor = 20
    for key in list(data):
        run, category, taskName, subject = key
        value = data[key]
        newValue = [0] * factor
        for i, voxel in enumerate(value):
            newValue[(i * factor) / Constants.NUM_VOXELS] += voxel
        data[key] = newValue

    return data


def dataSmoothing(data):
    numFeaturesSmoothed = 50
    for key in list(data):
        run, category, taskName, subject = key
        values = data[key]
        newValues = list(values)
        for i in range(Constants.NUM_VOXELS):
            count = 1
            for j in range(1, numFeaturesSmoothed):
                front = i - j
                back = i + j
                if front >= 0:
                    newValues[i] += float(values[front]) / j
                    count += 1
                if back < Constants.NUM_VOXELS:
                    newValues[i] += float(values[back]) / j
                    count += 1
            newValues[i] /= count
        data[key] = newValues
    return data


'''
Return a map of the data where each key is the  (run, imageType, taskName, subjectNum)
 and the value is the voxel array for the given (runNum, category, task).
'''


def getVoxelArray(allTasks=True, includeOddBall=False, includeWorkingMemory=False, includeSelectiveAttention=True,
                  subjectNumbers=[1, 2, 3, 4]):
    arr = []
    # for person 1
    if containsOne(subjectNumbers):
        mat = scipy.io.loadmat('sub1_betas.mat')
        oddBall = (mat['OB'], 'oddBall', 's1')
        workingMemory = (mat['WM'], 'workingMemory', 's1')
        selectiveAttention = (mat['SA'], 'selectiveAttention', 's1')
        arr.append(oddBall)
        arr.append(workingMemory)
        arr.append(selectiveAttention)

    # mat2 = scipy.io.loadmat('../data/sub2_betas.mat')
    # dataFiles = ['sub2_betas', 'sub3_betas', 'sub4_betas']
    for i in subjectNumbers:
        if i != 1:
            fileName = 'sub' + str(i) + '_betas'
            threeTasksData = parse(fileName, 's' + str(i))
            for task in threeTasksData:
                arr.append(task)
                # vA, taskType, SubjectNumber
                # VA is formatted as a 3d matrix where voxelAmplitude = task[vNum][imageType][run]
                ## want to get data into a voxel, category, runNumber
                # right now we have it voxel, condition, runNumber

                # amplitude = M[vNum][imageTupe][run]
    if allTasks == False:
        allTask = []
        for j in range(len(arr)):
            if includeOddBall:
                if arr[j][1] == 'oddBall':
                    allTask.append(arr[j])
            if includeWorkingMemory:
                if arr[j][1] == 'workingMemory':
                    allTask.append(arr[j])
            if includeSelectiveAttention:
                if arr[j][1] == 'selectiveAttention':
                    allTask.append(arr[j])
        arr = allTask


    voxelArrayMap = {}

    for task in arr:
        taskMatrix = task[0]
        taskName = task[1]
        subjectNum = task[2]
        for run in range(Constants.NUM_RUNS):
            # iterate through each category
            for imageType in range(Constants.NUM_CATEGORIES):
                voxelAmpsAtCategory = []
                # for each voxel in that particular category, get the value at [voxel][category][iteration]
                for vNum in range(1973):
                    voxelAmpsAtCategory.append(taskMatrix[vNum][imageType][run])
                key = (run, imageType, taskName, subjectNum)  # y = category
                voxelArrayMap[key] = voxelAmpsAtCategory
    return voxelArrayMap