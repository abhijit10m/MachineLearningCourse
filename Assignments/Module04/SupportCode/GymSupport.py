import math
import logging
import sys

# format = "%(asctime)s: %(message)s"
# logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')


# This code converts the continuous space states returned from Gym into discrete states so we can implement Q-Learning with tables.

def _DimensionBinIndex(binMin, binMax, binsPerDimension, value):
    if value < binMin:
        return 0

    if value > binMax:
        return binsPerDimension - 1

    range = binMax - binMin
    binWidth = range / float(binsPerDimension)
    binIndex = math.floor((value - binMin) / binWidth)

    return min(binIndex, binsPerDimension - 1) # this min is so the max value ends up in the last bin, not beyond it

class ContinuousToDiscrete(object):
    def __init__(self, binsPerDimension, dimensionLowerBounds, dimensionUpperBounds):
        logger.debug("%s, %s, %s", binsPerDimension, dimensionLowerBounds, dimensionUpperBounds)
        self.binsPerDimension = binsPerDimension
        self.dimensionLowerBounds = dimensionLowerBounds 
        self.dimensionUpperBounds = dimensionUpperBounds 

    def StateSpaceShape(self):
        return [ self.binsPerDimension, self.binsPerDimension, self.binsPerDimension, self.binsPerDimension ]

    def Convert(self, continuousObservation):
        return [ _DimensionBinIndex(self.dimensionLowerBounds[i], self.dimensionUpperBounds[i], self.binsPerDimension, continuousObservation[i]) for i in range(len(self.dimensionLowerBounds)) ]
