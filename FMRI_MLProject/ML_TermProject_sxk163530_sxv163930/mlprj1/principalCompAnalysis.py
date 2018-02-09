import numpy
from numpy import linalg
import util
import Constants


def PCA(normalizedData=None):
    if normalizedData is None:
        normalizedData = util.getVoxelArray(allTasks=True)

    # prepocess and normalize data
    util.normalize(normalizedData)
    m = len(normalizedData)

    print('Calculating the Covariance Matrix')

    count = 0
    # compute Covariance matrix [1/m Sum (x xT)]
    covarianceMatrix = [[0 for _ in range(Constants.NUM_VOXELS)] for _ in range(Constants.NUM_VOXELS)]
    for key in normalizedData:
        x = normalizedData[key]
        for i in range(Constants.NUM_VOXELS):
            for j in range(Constants.NUM_VOXELS):
                if count % 50000000 is 0:
                    print(int(float(count) * 100 / (m * Constants.NUM_VOXELS * Constants.NUM_VOXELS)), 'percent completed')
                covarianceMatrix[i][j] += (x[i] * x[j]) / m
                count += 1

    for i in range(Constants.NUM_VOXELS):
        for j in range(Constants.NUM_VOXELS):
            if isinstance(covarianceMatrix[i][j], complex):
                print('Covariance Matrix index', i, j, 'is complex:', covarianceMatrix[i][j])
    print('Finding the', Constants.K_DIMENSIONS, 'Largest Eigenvalues')
    # then compute all eigenvectors and corresponding eigenvalues
    eigenvalues, eigenvectors = linalg.eig(covarianceMatrix)
    eigens = zip(eigenvalues, eigenvectors)
    eigens.sort(key=lambda x: x[0], reverse=True)
    eigens = eigens[0:Constants.K_DIMENSIONS]
    eigenvalues, eigenvectors = zip(*eigens)

    for i in range(Constants.K_DIMENSIONS):
        print (i, eigenvalues[i])

    # transform data into K dimensions
    for key in normalizedData:
        x = normalizedData[key]
        y = [sum([eigenvector[i] * x[i] for i in range(Constants.NUM_VOXELS)])\
             for index, eigenvector in enumerate(eigenvectors)]
        normalizedData[key] = y