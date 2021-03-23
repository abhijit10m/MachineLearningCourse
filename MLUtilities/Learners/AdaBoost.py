import collections
from collections import defaultdict 
import math
import time
import copy
import numpy as np
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module2/Assignment4/"

## To save on time in boosted trees I reduced the runtime by avoiding prediction on every round.
#  I just predicted once, and then evaluate on different classification thresholds
class Hypothesis(object):
    def __init__(self, h, logibeta):
        self.h = h
        self.logibeta = logibeta
        

    def predict(self, x, classificationThreshold=0.5):

        startTime = time.time()
        numberOfExamples = len(x)
        # print("Boosted trees with %d rounds begining prediction of %d examples at (%.2f seconds)" % (len(self.h), numberOfExamples, startTime))

        predictions = np.array([(self.h[i].predict(x)) for i in range(len(self.h))]).transpose()
        predictionSum = [defaultdict(float) for i in range(numberOfExamples)]

        for i in range(numberOfExamples):
            predictionSum[i][0] =  math.fsum([ self.logibeta[k] for k in np.where(predictions[i][:] == 0)[0] ] )
            predictionSum[i][1] =  math.fsum([ self.logibeta[k] for k in np.where(predictions[i][:] == 1)[0] ] )

        yProbability = np.array([    (( float( predictionSum[i][1] ) + 1) / ( float(sum(predictionSum[i].values())) + 2) )     for i in range(numberOfExamples) ])
        yPredict = [ 1 if yProbability[i] > classificationThreshold else 0 for i in range(numberOfExamples)]

        endTime = time.time()
        runtime = endTime - startTime

        # print("Boosted trees with %d rounds predicted %d examples in (%.2f seconds)" % (len(self.h), numberOfExamples, runtime))

        return yPredict

## Optimized to save runtime when sweeping rounds parameter.
    def getModelWithRounds(self, rounds):
        return Hypothesis(self.h[:rounds], self.logibeta[:rounds]) 

class AdaBoost(object):

    def __init__(self, k, model, maxDepth=10, classificationThreshold = 0.5):
        self.isInitialized = False

        if k != None:
            self.__initialize(k, model, maxDepth, classificationThreshold)


    def __initialize(self, k, model, maxDepth, classificationThreshold):
        self.k = k
        self.w = []
        self.model = model
        self.maxDepth = maxDepth
        self.classificationThreshold = classificationThreshold
        self.isInitialized = True

    def adaBoost(self, x, y):

        h = []
        logibeta = []
        w0 = [1/len(y) for i in y]
        self.w.append(w0)
        sampleCount = len(y)

        for r in range(self.k):
            print("round {}".format(r))

            p = [ (self.w[r][i]) / (sum(self.w[r])) for i in range(len(y))]

            h0 = self.model()

            h0.fit(x, y, p, self.maxDepth, verbose=True)
            h0.visualize()

            yPredict = h0.predict(x, self.classificationThreshold)

            def notCorrect(y_p, y_a):
                if y_p == y_a:
                    return 0
                return 1

            error = [ p[i] * notCorrect(yPredict[i], y[i]) for i in range(sampleCount)]

            beta0 = sum(error) / (1 - sum(error))
            w1 =   [ self.w[r][i] * math.pow(beta0, 1 - notCorrect(yPredict[i], y[i]))   for i in range(sampleCount)]

            if sum(w1) == 0:
                print("All Correct")
                self.k = r - 1
                break

            if sum(error) > 0.5:
                print("hopeless : {}".format(sum(error)))
                self.k = r - 1
                break

            self.w.append(w1)
            logibeta.append(math.log(1/beta0))
            h.append(h0)

        return Hypothesis(h, logibeta)