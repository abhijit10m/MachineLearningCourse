# from google.colab import drive

import os
import sys
from PIL import Image
import torch 
from torch.utils import data
import time
import math
import logging
import itertools
from torchvision import models
from joblib import Parallel, delayed
import multiprocessing.dummy as mp
import numpy as np
from itertools import repeat
from functools import cmp_to_key

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')

confidenceIntervals = [.5, .8, .9, .95, .99]

bestParameters = dict()

bestParameters['lr'] = 1.0
bestParameters['convergence'] = 0.01
bestParameters['spec'] = None
bestParameters['accuracy'] = 0


## {'c1': {'in_channels': 576, 'stride': 1, 'kernel': 1}, 'c2': {'stride': 1, 'kernel': 1}, 'f1': {'out_channles': 128}, 'f2': {'out_channles': 16}}
class SpecFactory(object):

    def __init__(self):
        self.convulationLayer1InChannels = 24 * 24
        self.convulationLayer1Stride = [1]
        self.convulationLayer1Kernel = [24 * 24]
        self.convulationLayer1Specs = list(itertools.product(self.convulationLayer1Stride, self.convulationLayer1Kernel))
        self.convulationLayer2Stride = [1]
        self.convulationLayer2kernel = [12 * 12]
        self.convulationLayer2Specs = list(itertools.product(self.convulationLayer1Stride, self.convulationLayer1Kernel))
        self.fullyConnectedLayer1OutChannels = [5, 10, 15, 20, 25, 30]
        self.fullyConnectedLayer2OutChannels = [2, 5, 8, 10, 12, 15, 18, 20]
        self.learningRate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        self.convergence = [0.001, 0.0001, 0.00001]

    def __createFromSpecInfo(self, specInfo):
        return {
            "learningRate" : 0.0001,
            "convergence" : 0.0001,
            "c1" : {
                "in_channels" : self.convulationLayer1InChannels,
                "stride" : specInfo[0][0],
                "kernel" : specInfo[0][1]
            },
            "c2" : {
                "stride" : specInfo[1][0],
                "kernel" : specInfo[1][1]
            },
            "f1" : {
                "out_channles" : specInfo[2]
            },
            "f2" : {
                "out_channles" : specInfo[3]
            }
        }

    def create(self):
        all_specs = list(itertools.product(self.convulationLayer1Specs, self.convulationLayer2Specs, self.fullyConnectedLayer1OutChannels, self.fullyConnectedLayer2OutChannels))
        return [
            self.__createFromSpecInfo(specInfo) for specInfo in all_specs
        ]


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    global evaluateFold

    startTime = time.time()

    numberOfFolds = 2
    batchSize = 100
    numberOfCorrects = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device is: %s", device)

    def evaluateFold(foldId):
        logger.info("evaluating fold %s", foldId)
        numberOfCorrect = 0
        (xTrainRawK, yTrainK, xEvaluateRawK, yEvaluateK) = CrossValidation.CrossValidation(xTrainRaw, yTrain, numberOfFolds, foldId)

        trainDataSet = EyeDataset.EyeDataset(xTrainRawK, yTrainK)
        evaluateDataSet = EyeDataset.EyeDataset(xEvaluateRawK, yEvaluateK)

        xTrainDataSetGenerator = data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True, num_workers=5)
        xEvaluateDataSetEvaluation = data.DataLoader(evaluateDataSet, batch_size=len(xEvaluateRawK), shuffle=True, num_workers=10)

        lossFunction = torch.nn.BCELoss()
        model = BlinkNeuralNetwork.BlinkNeuralNetwork(runSpecification)
        optimizer = torch.optim.SGD(model.parameters(), lr=runSpecification['learningRate'])
        model.to(device)
        logger.info("training")
        converged = False
        epoch = 0
        while not converged and epoch < 5000:

            model.train(mode=True)
            count = 0
            for (batchXTensor, batchYTensor) in xTrainDataSetGenerator:
                logger.info('%d, %d', count, len(batchXTensor))
                count += 1

                batchXTensorGPU = batchXTensor.to(device)
                batchYTensorGPU = batchYTensor.to(device)

                # Do the forward pass
                yTrainPredicted = model(batchXTensorGPU)

                # Compute the total loss summed across training samples in the epoch
                #  note this is different from our implementation, which took one step
                #  of gradient descent per sample.
                loss = lossFunction(yTrainPredicted, batchYTensorGPU)

                optimizer.zero_grad()

                # Backprop the errors from the loss on this iteration
                loss.backward()

                # Do a weight update step
                optimizer.step()

            epoch = epoch + 1

        model.train(mode = False)
        logger.info("cross validating")
        yEvaluatePredicted = None
        yEvaluateActual = None
        print(xEvaluateDataSetEvaluation)
        for (batchXTensor, batchYTensor) in xEvaluateDataSetEvaluation:
            # Do the forward pass
            yEvaluatePredicted = model(batchXTensor.to(device))
            yEvaluateActual = batchYTensor.to(device)
        numberOfCorrect = 0
        for i in range(len(yEvaluatePredicted)):
            if yEvaluateActual[i] == yEvaluatePredicted[i]:
                numberOfCorrect += 1

        runSpecification['model'] = dict()
        runSpecification['model'][foldId] = model
        # numberOfCorrects.append(numberOfCorrect)
        return numberOfCorrect

    with mp.Pool() as p:
        numberOfCorrects = p.map(evaluateFold, range(numberOfFolds))
    p.close()
    p.join()


    # HERE upgrade this to calculate and save some error bounds...
    fold_accuracy = sum(numberOfCorrects) / len(xTrainRaw)
    runSpecification['accuracy'] = fold_accuracy

    runSpecification['confidenceIntervals'] = dict()
    for confidence in confidenceIntervals:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(fold_accuracy, len(xTrainRaw), confidence)
        runSpecification['confidenceIntervals'][confidence] = dict()
        runSpecification['confidenceIntervals'][confidence]['lowerBound'] = lowerBound
        runSpecification['confidenceIntervals'][confidence]['upperBound'] = upperBound
    
    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime
    return runSpecification

if __name__ == '__main__':

    # drive.mount
    # drive.mount('/content/gdrive', force_remount=True)

    # sys.path.append('/content/gdrive/MyDrive/ColabNotebooks')
    # os.chdir('/content/gdrive/MyDrive/ColabNotebooks')
    # kDataPath = os.path.join( "MachineLearningCourse", "MLProjectSupport", "Blink", "dataset")
    # print(kDataPath)
    # kOutputDirectory = "MachineLearningCourse/visualize/Module3/Assignment4"

    # kDataPath = os.path.join( "MachineLearningCourse", "MLProjectSupport", "Blink", "dataset")
    # print(kDataPath)
    import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    import MachineLearningCourse.Assignments.Module03.SupportCode.EyeDataset as EyeDataset
    import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetworkOptimized as BlinkNeuralNetwork
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds


    (xRaw, yRaw) = BlinkDataset.LoadRawData()

    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate = .1, percentTest = .1)

    print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
    print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
    print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

    torch.manual_seed(0)

    runSpecs = SpecFactory().create()

    evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in runSpecs)

    print(evaluations)
