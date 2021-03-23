import random
import math
import time
import numpy as np
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate
import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

class NeuralNetworkFullyConnected(object):
    """Framework for fully connected neural network"""
    def __init__(self, numInputFeatures, hiddenLayersNodeCounts=[2], seed=1000):

        np.random.seed(seed)
        self.totalEpochs = 0
        self.lastLoss    = None
        self.converged   = False


        self.layers = []
        self.activations = []
        self.errors = []

        # set up the input layer
        self.layerWidths = [ numInputFeatures ]
        self.activations.append(np.zeros(numInputFeatures))

        # set up the hidden layers
        for i in range(len(hiddenLayersNodeCounts)):
            self.layerWidths.append(hiddenLayersNodeCounts[i])

        # output layer
        self.layerWidths.append(1)
        self.layerWidths = np.asarray(self.layerWidths)

        ###
        ## now set up all the parameters and any arrays you want for forward/backpropagation
        ###

        for i in range(1, len(self.layerWidths)):
            
            inputCounts = self.layerWidths[i-1]
            
            stdv = 1 / (math.sqrt( inputCounts + 1 ))

            activations = np.zeros(shape=(self.layerWidths[i]))
            errors = np.zeros(shape=(self.layerWidths[i]))

            self.activations.append(activations)
            self.errors.append(errors)

            layerWeights = []

            for nodeCount in range(self.layerWidths[i]):
                nodeWeights = np.random.uniform(-1 * stdv, stdv, inputCounts + 1)
                layerWeights.append(nodeWeights)

            self.layers.append(np.asarray(layerWeights))

        self.activations = np.asarray(self.activations)
        self.errors = np.asarray(self.errors)
        self.layers = np.asarray(self.layers, dtype = object)

    def sigmoid(self, score):
        return 1 / (1 + math.exp(-1 * score))

    def feedForward(self, x):
        np.copyto(self.activations[0], x)
        for i in range(0, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.activations[i+1][j] =  self.sigmoid(np.dot(np.append(np.asarray([1.0]), self.activations[i]), self.layers[i][j]))


    def backpropagate(self, y):
        yPredict = self.activations[-1][0]
        outputError = yPredict * (1 - yPredict) * (y - yPredict)
        self.errors[-1][0] = outputError

        for i in reversed(range(len(self.layers) - 1)):
            logging.debug("layers[%d] %s", i+1, self.layers[i+1])
            for j in range(len(self.layers[i])):
                sumOfErrors = 0

                for k in range(len(self.layers[i+1])):
                    logging.debug("errors[%d][%d] %s", i+1, k, self.errors[i+1][k])
                    sumOfErrors += self.errors[i+1][k] * self.layers[i+1][k][j+1]

                if sumOfErrors == 0:
                    logging.info('sumOfErrors 0 %f', sumOfErrors)

                self.errors[i][j] = self.activations[i+1][j] * (1 - self.activations[i+1][j]) * sumOfErrors

        logging.debug(self.errors)

    def updateweights(self, stepSize, momentum=None):

        for i in reversed(range(len(self.layers))):
            for j in range(len(self.layers[i])):
                for k in range(1, len(self.layers[i][j])):

                    if self.errors[i][j] == 0:
                        logging.debug("errors 0")
                        logging.debug("i: %d, j: %d, errors: %s", i, j, self.errors)

                    if self.activations[i][k - 1] == 0:
                        logging.info("activations 0")
                        logging.info("i: %d, k: %d, activations: %s", i, k-1, self.activations)

                    self.layers[i][j][k] += stepSize * self.errors[i][j] * self.activations[i][k - 1]
                self.layers[i][j][0] += stepSize * self.errors[i][j]

    def loss(self, x, y):        
        return EvaluateBinaryProbabilityEstimate.MeanSquaredErrorLoss(y, self.predictProbabilities(x))

    def predictOneProbability(self, x):
        self.feedForward(x)
        # return the activation of the neuron in the output layer
        return self.activations[-1][0]

    def predictProbabilities(self, x):    
        return [ self.predictOneProbability(sample) for sample in x ]

    def predict(self, x, threshold = 0.5):
        return [ 1 if probability > threshold else 0 for probability in self.predictProbabilities(x) ]

    def __CheckForConvergence(self, x, y, convergence):
        loss = self.loss(x,y)

        if self.lastLoss != None:
            deltaLoss = abs(self.lastLoss - loss)
            self.converged = deltaLoss < convergence
        self.lastLoss = loss
    
    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, maxEpochs=1, stepSize=0.01, convergence = None):
        for _ in range(maxEpochs):
            if self.converged:
                return
            # do a full epoch of stocastic gradient descent
            for (example, output)  in zip(x,y):
                self.feedForward(example)
                self.backpropagate(output)
                self.updateweights(stepSize, momentum=None)
            
            self.totalEpochs += 1
            
            if convergence != None:
                self.__CheckForConvergence(x, y, convergence)
             
                
    def fit(self, x, y, maxEpochs=50000, stepSize=0.01, convergence=0.00001, verbose = True):        
        startTime = time.time()
        
        self.incrementalFit(x, y, maxEpochs=maxEpochs, stepSize=stepSize, convergence=convergence)
        
        endTime = time.time()
        runtime = endTime - startTime
      
        if not self.converged:
            logging.warning("Warning: NeuralNetwork did not converge after the maximum allowed number of epochs.")
        elif verbose:
            logging.info("NeuralNetwork converged in %d epochs (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f.", self.totalEpochs, runtime, len(x[0]), stepSize, convergence)

