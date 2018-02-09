import pylab
import util
import numpy

class Grapher:
  def twoDGraph(self, data):
    count = 0
    for key in data:
      fig = pylab.figure()
      values = data[key]
      pylab.plot(values)
      print (key)
      count += 1
      if count == 2:
        break
    pylab.show()

  def twoDHeatMap(self, data, vmin=None, vmax=None, title=None, xLabel=None, yLabel=None, filename=None):
    if not vmin is None and not vmax is None:
      pylab.pcolor(numpy.array(data), vmin=vmin, vmax=vmax)
    else:
      pylab.pcolor(numpy.array(data))
    if not title is None:
      pylab.title(title)
    if not xLabel is None:
      pylab.xlabel(xLabel)
    if not yLabel is None:
      pylab.ylabel(yLabel)
    if not filename is None:
      pylab.savefig(filename + '.png')
    pylab.show()

if __name__ == '__main__':
  data = util.getVoxelArray()
  data = util.dataSmoothing(data)
  util.normalize(data)
  g = Grapher()
  g.twoDGraph(data)