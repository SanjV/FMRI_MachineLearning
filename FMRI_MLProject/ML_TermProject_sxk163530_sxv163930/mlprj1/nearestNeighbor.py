import random
import math
import util
import principalCompAnalysis
import numpy
import Constants


class NearestNeighbor:
  def __init__(self, data=None, normalize=True, categories=[0, 1, 2, 3, 4], subjects=[1, 2, 3, 4]):
    if data is None:
      data = util.getVoxelArray(subjectNumbers=subjects)
    if normalize:
      util.normalize(data)
    if len(categories) < Constants.NUM_CATEGORIES:
      util.filterData(data, categories)
    self.data = data
    self.categories = categories

  def kNearestNeighbor(self, k=1):
    # Choose one to leave out, out of all
    totalError = 0
    totalPredictions = len(self.data)

    for key in self.data:
      testedExample = self.data[key]
      self.data.pop(key, None)

      # (distance, category)
      nearestNeighbors = []
      for neighbor in self.data:
        category = neighbor[1]
        distance = util.getDistance(self.data[neighbor], testedExample)
        if len(nearestNeighbors) < k:
          nearestNeighbors.append((distance, category))
          nearestNeighbors.sort()
        elif distance < nearestNeighbors[k - 1][0]:
          nearestNeighbors[k - 1] = (distance, category)
          nearestNeighbors.sort()

      closestNeighbors = [0] * Constants.NUM_CATEGORIES
      for neighbor in nearestNeighbors:
        distance, category = neighbor
        closestNeighbors[category] += 1 / distance

      c = sum(closestNeighbors)
      closestNeighbors = [x / c for x in closestNeighbors]

      choice = numpy.random.choice(range(Constants.NUM_CATEGORIES), p=closestNeighbors)

      if key[1] != choice:
        # print ("Tested on category ", key[1], " Classified it as: ", choice)
        totalError +=1
      self.data[key] = testedExample
    percentCorrect = float(totalPredictions - totalError) * 100 / totalPredictions
    print (self.categories, totalPredictions - totalError, 'out of', totalPredictions, 'correct', 'for k =', k, 'or', percentCorrect, '%')
    return percentCorrect

  def nearestNeighbor(self):
    return self.kNearestNeighbor(1)

def run(categories, subjects):
  data = util.getVoxelArray(subjectNumbers=subjects)
  nn = NearestNeighbor(data=data, categories=categories)
  return nn.kNearestNeighbor(5)

if __name__ == '__main__':
  print ('NEAREST NEIGHBOR BINARY CLASSIFICATION')
  # for i in range(Constants.NUM_SUBJECTS):
  classification = 0
  for i in range(Constants.NUM_CATEGORIES):
    for j in range(Constants.NUM_CATEGORIES - i - 1):
      classification += run([i, j + i + 1], [1, 2, 3, 4])
  print ('Average classification rate =', classification / ((Constants.NUM_CATEGORIES * (Constants.NUM_CATEGORIES - 1)) / 2))
