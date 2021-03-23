import MachineLearningCourse.MLUtilities.Image.Convolution2D as Convolution2D
from PIL import Image

import time
from joblib import Parallel, delayed
import numpy as np
import statistics

class BlinkFeaturize(object):
    def __init__(self):
        self.featureSetCreated = False
    
    def CreateFeatureSet(self, xRaw, yRaw, includeEdgeFeatures=True, includeRawPixels=False, includeIntensities=False, intensitiesSampleStride = 2, splitGrid3By3=False):
        self.includeEdgeFeatures = includeEdgeFeatures
        self.includeRawPixels = includeRawPixels
        self.includeIntensities = includeIntensities
        self.intensitiesSampleStride = int(intensitiesSampleStride)
        self.splitGrid3By3 = splitGrid3By3
        self.featureSetCreated = True
        
    def _FeaturizeX(self, xRaw):
        featureVector = []

        image = Image.open(xRaw)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if self.includeEdgeFeatures:
            yEdges = np.array(Convolution2D.Convolution3x3(image, Convolution2D.SobelY))
            xEdges = np.array(Convolution2D.Convolution3x3(image, Convolution2D.SobelX))

            avgYEdge = sum([sum([abs(value) for value in row]) for row in yEdges]) / numPixels
            avgXEdge = sum([sum([abs(value) for value in row]) for row in xEdges]) / numPixels
        
            featureVector.append(avgYEdge)
            featureVector.append(avgXEdge)

            if self.splitGrid3By3:

                xIntervals = np.arange(0, xSize + 1, xSize/3, dtype=np.int16)
                yIntervals = np.arange(0, ySize + 1, xSize/3, dtype=np.int16)

                for i in range(1, len(xIntervals)):
                    for j in range(1, len(yIntervals)):

                        x_selected = np.absolute(xEdges[xIntervals[i-1] : xIntervals[i], yIntervals[j-1] : yIntervals[j]])
                        y_selected = np.absolute(yEdges[xIntervals[i-1] : xIntervals[i], yIntervals[j-1] : yIntervals[j]])

                        featureVector.append(x_selected.min())
                        featureVector.append(y_selected.min())
                        featureVector.append(x_selected.max())
                        featureVector.append(y_selected.max())


        if self.includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    featureVector.append(pixels[x,y])

        if self.includeIntensities:
            for x in range(0, xSize, self.intensitiesSampleStride):
                for y in range(0, ySize, self.intensitiesSampleStride):
                    featureVector.append(pixels[x,y]/255.0)

        return featureVector

    def Featurize(self, xSetRaw, normalize = False, verbose = True):
        if not self.featureSetCreated:
            raise UserWarning("Trying to featurize before calling CreateFeatureSet")
        
        if verbose:
            print("Loading & featurizing %d image files..." % (len(xSetRaw)))
        
        
        startTime = time.time()

        # If you don't have joblib installed you can swap these comments
        # result = [ self._FeaturizeX(x) for x in xSetRaw ]

        result = Parallel(n_jobs=12)(delayed(self._FeaturizeX)(x) for x in xSetRaw)

        if normalize:
            result = np.asarray(result)
            numberOfFeatures = len(result[0])
            column_means = np.outer(np.ones(np.shape(result)[0]), np.mean(result, axis=0))
            column_std_dev = np.std(result, axis=0)
            result = result - column_means
            result = result / column_std_dev
            result = list(result)

        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("   Complete in %.2f seconds" % (runtime))
        
        return result
