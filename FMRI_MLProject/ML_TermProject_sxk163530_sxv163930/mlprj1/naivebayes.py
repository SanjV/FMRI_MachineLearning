import util
import Constants
import math
import principalCompAnalysis
import grapher


class NaiveBayes:
  def __init__(self, data=None, normalize=True, numTrainingExamples=24, categories=[0, 1, 2, 3, 4], subjects=[1, 2, 3, 4]):
    if data is None:
      data = util.getVoxelArray(subjectNumbers=subjects)
    if normalize:
      util.normalize(data)
    if len(categories) < Constants.NUM_CATEGORIES:
      util.filterData(data, categories)

    self.numCategories = Constants.NUM_CATEGORIES
    self.categories = None
    if not categories is None:
      self.numCategories = len(categories)
      self.categories = categories

    self.numTrainingExamples = numTrainingExamples * len(subjects) / Constants.NUM_SUBJECTS
    self.numTestingExamples = len(subjects) * Constants.NUM_RUNS * Constants.NUM_TASKS - self.numTrainingExamples
    self.trainExamples = [[[0] for _ in range(int(self.numTrainingExamples))] for _ in range(int(self.numCategories))]
    self.testExamples = []
    self.averages = [[0 for _ in range(Constants.NUM_VOXELS)] for _ in range(self.numCategories)]

    for key in data:
      value = data[key]
      run, category, task, subject = key
      if not self.categories is None:
        if category not in self.categories: continue
        category = self.getCategory(category)
      index = self.index(run, task, subject) * len(subjects) / Constants.NUM_SUBJECTS
      if index < self.numTrainingExamples:
        self.trainExamples[int(category)][int(index)] = list(value)
      else:
        self.testExamples.append((key, value))

  def train(self):
    counts = [[0 for _ in range(Constants.NUM_VOXELS)] for _ in range(self.numCategories)]
    for category in range(self.numCategories):
      for x in self.trainExamples[category]:
        for i, voxel in enumerate(x):
          counts[category][i] += voxel

    # divide by number of examples
    for category in range(self.numCategories):
      for i in range(Constants.NUM_VOXELS):
        self.averages[category][i] = counts[category][i] / float(Constants.NUM_VOXELS)

  # Test MLE Estimates b/c each p_y is equal
  def test(self):
    self.testResults = [[0 for _ in range(self.numCategories)] for _ in range(len(self.testExamples))]
    for category in range(self.numCategories):
      for testIndex, pair in enumerate(self.testExamples):
        key, value = pair
        probability = 0
        for i, voxel in enumerate(value):
          pdf = self.pdf(self.averages[category][i], 1, voxel)
          probability += math.log(pdf)
        self.testResults[testIndex][category] = probability

  # print test results
  def results(self):
    numTests = 0
    numErrors = 0
    for testIndex, pair in enumerate(self.testExamples):
      key, value = pair
      numTests += 1
      run, category, task, subject = key
      guess = self.testResults[testIndex].index(max(self.testResults[testIndex]))
      if not self.categories is None:
        guess = self.categories[guess]
      if not category is guess:
        numErrors += 1
        # print 'Incorrectly classified', key, 'as', guess, '...', self.testResults[testIndex]

    percentCorrect = float(numTests - numErrors) * 100 / numTests
    print (self.categories, '= Correctly classified:', numTests - numErrors, '/', \
      numTests, 'or', percentCorrect, '%')

    return percentCorrect

  def pdf(self, mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

  def taskToValue(self, task):
    if task is 'oddBall': return 0
    elif task is 'selectiveAttention': return 1
    elif task is 'workingMemory': return 2
    else: raise Exception('Wrong task name')

  # return associated index for run, task, subject
  def index(self, run, task, subject):
    task = self.taskToValue(task)
    subject = int(''.join(x for x in subject if str.isdigit(x)))
    return ((run + 2) % 3) * 12 + task * 4 + subject - 1

  def altIndex(self, run, task, subject):
    task = self.taskToValue(task)
    subject = int(''.join(x for x in subject if str.isdigit(x)))
    return task * 12 + run * 4 + subject - 1

  def leaveOneSubjectOutIndex(self, run, task, subject):
    task = self.taskToValue(task)
    subject = int(''.join(x for x in subject if str.isdigit(x)))
    return ((subject + 4) % 4) * 9 + run * 3 + task

  def getCategory(self, categoryValue):
    for i, category in enumerate(self.categories):
      if category == categoryValue:
        return i
    return -1

def run(categories=None, subject=None):
  data = util.getVoxelArray(subjectNumbers=subject)
  nb = NaiveBayes(data=data, categories=categories, subjects=subject)
  nb.train()
  nb.test()
  return nb.results()

if __name__ == '__main__':
  print ('NAIVE BAYES BINARY CLASSIFICATION')
  classification = 0
  for i in range(Constants.NUM_CATEGORIES):
    for j in range(Constants.NUM_CATEGORIES - i - 1):
      classification += run(categories=[i, j + i + 1], subject=[1, 2, 3, 4])
  print ('Average classification rate for =', classification / ((Constants.NUM_CATEGORIES * (Constants.NUM_CATEGORIES - 1)) / 2))

