kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module3/Assignment4/"

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
import sys
import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module03.SupportCode.EyeDataset as EyeDataset
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification



format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')

def runTrainMaxEpochs(model, lr, maxEpochs, number, xTrainDataSetGenerator, xTrainDataSetEvaluation):

    batchSize = len(xTrainRaw)
    numberOfCorrects = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device is: %s", device)

    losses = np.zeros([maxEpochs, 2])

    lossFunction = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    converged = False
    epoch = 0
    lastValidationLoss = None
    print("starting training")
    while not converged and epoch < maxEpochs:

        model.train(mode=True)

        for (batchXTensor, batchYTensor) in xTrainDataSetGenerator:

            batchXTensorGPU = batchXTensor.to(device)
            batchYTensorGPU = batchYTensor.to(device)

            yTrainPredicted = model(batchXTensorGPU)

            loss = lossFunction(yTrainPredicted, batchYTensorGPU)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        logger.info("epoch : %d", epoch)
        epoch = epoch + 1


    endTime = time.time()
    model.train(mode=False)

class BlinkNeuralNetworkColabModel1(torch.nn.Module):
    def __init__(self):
        super(BlinkNeuralNetworkColabModel1, self).__init__()
        self.a1 = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.f1 = torch.nn.Sequential(
            torch.nn.Linear(12 * 12, 7),
            torch.nn.ReLU())
        
        self.f2 = torch.nn.Sequential(
            torch.nn.Linear(7, 2),
            torch.nn.ReLU())
        
        self.o  = torch.nn.Sequential(
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid())


    def forward(self, x):

        out = self.a1(x)
        out = out.reshape(out.size(0), -1)
        out = self.f1(out)
        out = self.f2(out)
        out = self.o(out)
        return out




class BlinkNeuralNetworkColabModel2(torch.nn.Module):
    def __init__(self):
        super(BlinkNeuralNetworkColabModel2, self).__init__()

        self.b0 = torch.nn.BatchNorm2d(1)

        self.c1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 8, 3, 3),
                    torch.nn.ReLU())

        self.c2 = torch.nn.Sequential(
                    torch.nn.Conv2d(8, 4, 3, 3),
                    torch.nn.ReLU())

        self.b1 = torch.nn.BatchNorm1d(16)

        self.d1 = torch.nn.Dropout()

        self.f1 = torch.nn.Sequential(
                    torch.nn.Linear(16, 9),
                    torch.nn.ReLU())
        
        
        self.f2 = torch.nn.Sequential(
                    torch.nn.Linear(9, 2),
                    torch.nn.ReLU())

        self.o  = torch.nn.Sequential(
                    torch.nn.Linear(2, 1),
                    torch.nn.Sigmoid())



    def forward(self, x):
        out = self.b0(x)
        out = self.c1(out)  
        out = self.c2(out)  
        out = out.reshape(out.size(0), -1)
        out = self.d1(out)
        out = self.f1(out)
        out = self.f2(out)
        out = self.o(out)
        return out


class BlinkNeuralNetworkColabModel3(torch.nn.Module):
    def __init__(self):
        super(BlinkNeuralNetworkColabModel3, self).__init__()

        self.b0 = torch.nn.BatchNorm2d(1)

        self.c1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 8, 3, 3),
                    torch.nn.ReLU())

        self.c2 = torch.nn.Sequential(
                    torch.nn.Conv2d(8, 4, 3, 3),
                    torch.nn.ReLU())

        self.b1 = torch.nn.BatchNorm1d(16)

        self.d1 = torch.nn.Dropout()

        self.f1 = torch.nn.Sequential(
                    torch.nn.Linear(16, 9),
                    torch.nn.ReLU())

        self.d2 = torch.nn.Dropout()

        self.f2 = torch.nn.Sequential(
                    torch.nn.Linear(9, 2),
                    torch.nn.ReLU())

        self.o  = torch.nn.Sequential(
                    torch.nn.Linear(2, 1),
                    torch.nn.Sigmoid())



    def forward(self, x):
        logger.debug(x.size())
        out = self.b0(x)
        out = self.c1(out)
        out = self.c2(out)  
        out = out.reshape(out.size(0), -1)
        out = self.d1(out)
        out = self.f1(out)
        out = self.d2(out)
        out = self.f2(out)
        out = self.o(out)
        return out

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xTestDataSetEvaluation):
    global f
    model.train(mode=False)
    pointsToEvaluate = 100
    thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPR_dict = dict()
    FNR_dict = dict()

    FPRs = []
    FNRs = []
    xTestDataSetEvaluation = xTestDataSetEvaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def f(threshold):
        try:
            yTestActual = None
            yTestProbability = None
            yTestEvaluation = None
            for (batchXTensor, batchYTensor) in xTestDataSetEvaluation:
                batchXTensorGPU = batchXTensor.to(device)
                batchYTensorGPU = batchYTensor.to(device)
                yTestActual = batchYTensorGPU
                yTestProbability = model(batchXTensorGPU)
                yTestEvaluation = [ 1 if pred > threshold else 0 for pred in yTestProbability ]

            FPR = EvaluateBinaryClassification.FalsePositiveRate(yTestActual, yTestEvaluation)
            FNR = EvaluateBinaryClassification.FalseNegativeRate(yTestActual, yTestEvaluation)
            logging.debug("threshold : %s, FPR: %s, FNR: %s", threshold, FPR, FNR)
        except NotImplementedError:
            raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

        FPR_dict[threshold] = FPR
        FNR_dict[threshold] = FNR
        return (threshold, FPR, FNR)


    try:
        with mp.Pool() as p:
            p.map(f, thresholds)
        p.close()
        p.join()
        # for t in thresholds:
        #     f(t)

        for threshold in thresholds:
            FPRs.append(FPR_dict[threshold])
            FNRs.append(FNR_dict[threshold])
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)


def trainAndEvalute(model, lr, maxEpochs, id, xTrainDataSetGenerator, xTrainDataSetEvaluation, name):
        runTrainMaxEpochs(model, lr, maxEpochs, id, xTrainDataSetGenerator, xTrainDataSetEvaluation)
        torch.save(model.state_dict(), "{}/{}".format(kOutputDirectory, name))


if __name__ == "__main__":

    (xRaw, yRaw) = BlinkDataset.LoadRawData()

    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate = .1, percentTest = .1)

    print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
    print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
    print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

    torch.manual_seed(0)

    trainDataSet = EyeDataset.EyeDataset(xTrainRaw, yTrain)
    validateDataSet = EyeDataset.EyeDataset(xValidateRaw, yValidate)
    testDataSet = EyeDataset.EyeDataset(xTestRaw, yTest)

    xTrainDataSetGenerator = data.DataLoader(trainDataSet, batch_size=len(xTrainRaw), shuffle=True, num_workers=1)
    xTestDataSetEvaluation = data.DataLoader(testDataSet, batch_size=len(xTestRaw), shuffle=True, num_workers=1)
    xTrainDataSetEvaluation = data.DataLoader(trainDataSet, batch_size=len(xTrainRaw), shuffle=True, num_workers=1)
    xValidateDataSetEvaluation = data.DataLoader(validateDataSet, batch_size=len(xValidateRaw), shuffle=True, num_workers=1)

    model1 = BlinkNeuralNetworkColabModel1()
    model1.load_state_dict(torch.load("{}/{}".format(kOutputDirectory, "model1") ))
    model2 = BlinkNeuralNetworkColabModel2()
    model2.load_state_dict(torch.load("{}/{}".format(kOutputDirectory, "model2") ))
    # model3 = BlinkNeuralNetworkColabModel3()

    model1_lr = 0.01
    model2_lr = 0.01
    model3_lr =  0.05

    maxEpochs = 500

    # trainAndEvalute(model1, model1_lr, maxEpochs, 1, xTrainDataSetGenerator, xTrainDataSetEvaluation, "model1")

    seriesFPRs = []
    seriesFNRs = []

    (FPRs, ModelFNRs, thresholds) = TabulateModelPerformanceForROC(model1, xTestDataSetEvaluation)
    seriesFPRs.append(FPRs)
    seriesFNRs.append(ModelFNRs)

    # Charting.PlotROCs(seriesFPRs, seriesFNRs, ["Simple Model"], useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Blink-BestModel_1")


    # trainAndEvalute(model2, model2_lr, maxEpochs, 2, xTrainDataSetGenerator, xTrainDataSetEvaluation, "model2")

    (FPRs, ModelFNRs, thresholds) = TabulateModelPerformanceForROC(model2, xTestDataSetEvaluation)
    seriesFPRs.append(FPRs)
    seriesFNRs.append(ModelFNRs)

    Charting.PlotROCs(seriesFPRs, seriesFNRs, ["Simple Model", "Model with 2 Dropout"], useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Blink-BestModel_2")




    # trainAndEvalute(model3, model3_lr, maxEpochs, 3, xTrainDataSetGenerator, xTrainDataSetEvaluation, "model3")
    # (FPRs, ModelFNRs, thresholds) = TabulateModelPerformanceForROC(model3, xTestDataSetEvaluation)

    # seriesFPRs.append(FPRs)
    # seriesFNRs.append(ModelFNRs)

    # Charting.PlotROCs(seriesFPRs, seriesFNRs, ["Simple Model", "Model with 2 Dropout"], useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Blink-BestModel_3")
