kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module3/Assignment4"

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize
from PIL import Image
import torch 
from torch.utils import data
import EyeDataset
import time
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetworkOptimized as BlinkNeuralNetwork
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import itertools
import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


class SpecFactory(object):

    def __init__(self):
        self.convulationLayer1InChannels = 24 * 24
        self.convulationLayer1Stride = [1,2,3,4,5]
        self.convulationLayer1Kernel = [1,2,3,4,5]
        self.convulationLayer1Specs = list(itertools.product(self.convulationLayer1Stride, self.convulationLayer1Kernel))
        self.convulationLayer2Stride = [1,2,3,4,5]
        self.convulationLayer2kernel = [1,2,3,4,5]
        self.convulationLayer2Specs = list(itertools.product(self.convulationLayer1Stride, self.convulationLayer1Kernel))
        self.fullyConnectedLayer1OutChannels = [128, 96, 64, 32]
        self.fullyConnectedLayer2OutChannels = [16, 8, 4, 2]

    def __createFromSpecInfo(self, specInfo):
        return {
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


allSpecs = SpecFactory().create()
print(allSpecs[0])

if __name__ != '__main__':


    (xRaw, yRaw) = BlinkDataset.LoadRawData()

    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate = .1, percentTest = .1)

    print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
    print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
    print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

    torch.manual_seed(0)

    trainDataSet = EyeDataset.EyeDataset(xTrainRaw, yTrain)
    validateDataSet = EyeDataset.EyeDataset(xValidateRaw, yValidate)
    testDataSet = EyeDataset.EyeDataset(xTestRaw, yTest)

    batchSize = 100

    trainDataSetGenerator = data.DataLoader(trainDataSet, batch_size=batchSize, shuffle=True, num_workers=5)
    validateDataSetGenerator = data.DataLoader(validateDataSet, batch_size=batchSize, shuffle=True, num_workers=5)
    testDataSetGenerator = data.DataLoader(testDataSet, batch_size=batchSize, shuffle=True, num_workers=5)

    trainDataSetEvaluation = data.DataLoader(trainDataSet, batch_size=len(trainDataSet), shuffle=False, num_workers=10)
    validateDataSetEvaluation = data.DataLoader(validateDataSet, batch_size=len(validateDataSet), shuffle=False, num_workers=10)
    testDataSetEvaluation = data.DataLoader(testDataSet, batch_size=len(testDataSet), shuffle=False, num_workers=10)


    # Create the model
    model = BlinkNeuralNetwork.BlinkNeuralNetwork()

    # Create the loss function to use (Mean Square Error)
    # lossFunction = torch.nn.MSELoss(reduction='mean')
    lossFunction = torch.nn.BCELoss()

    # Create the optimization method (Stochastic Gradient Descent) and the step size (lr -> learning rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    ##
    # Move the model and data to the GPU if you're using your GPU
    ##

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)

    model.to(device)

    ##
    # Build the model
    ##
    startTime = time.time()

    trainLosses = []
    validationLosses = []
    testLosses = []

    converged = False
    epoch = 1
    lastValidationLoss = None

    while not converged and epoch < 5000:

        model.train(mode=True)

        for (batchXTensor, batchYTensor) in trainDataSetGenerator:

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

        # now check the validation loss
        model.train(mode=False)

        yTrainPredicted = None
        yTrainActual = None

        yValidatePredicted = None
        yValidateActual = None

        yTestPredicted = None
        yTestActual = None

        for (batchXTensor, batchYTensor) in trainDataSetEvaluation:
            # Do the forward pass
            yTrainPredicted = model(batchXTensor)
            yTrainActual = batchYTensor

        for (batchXTensor, batchYTensor) in validateDataSetEvaluation:
            # Do the forward pass
            yValidatePredicted = model(batchXTensor)
            yValidateActual = batchYTensor

        for (batchXTensor, batchYTensor) in testDataSetEvaluation:
            # Do the forward pass
            yTestPredicted = model(batchXTensor)
            yTestActual = batchYTensor

        # Compute the total loss summed across training samples in the epoch
        #  note this is different from our implementation, which took one step
        #  of gradient descent per sample.
        trainLoss = lossFunction(yTrainPredicted, yTrainActual)
        validationLoss = lossFunction(yValidatePredicted, yValidateActual)
        testLoss = lossFunction(yTestPredicted, yTestActual)

        trainLosses.append(trainLoss.item() / len(yTrainPredicted))

        validationLosses.append(validationLoss.item() / len(yValidatePredicted))

        testLosses.append(testLoss.item() / len(yTestPredicted))

        xValues   = [ i + 1 for i in range(len(trainLosses))]

        Charting.PlotSeries([trainLosses, validationLosses], ["Train Loss", "Validate Loss"], xValues, useMarkers=False, chartTitle="Pytorch Train vs Validate Run", xAxisTitle="Epoch", yAxisTitle="Loss", yBotLimit=0.0, outputDirectory=kOutputDirectory, fileName="Blink-TrainValidate")
        Charting.PlotSeries([trainLosses, validationLosses, testLosses], ["Train Loss", "Validate Loss", "Test Loss"], xValues, useMarkers=False, chartTitle="Pytorch Train vs Validate vs Test Run", xAxisTitle="Epoch", yAxisTitle="Loss", yBotLimit=0.0, outputDirectory=kOutputDirectory, fileName="Blink-TrainValidateTest")

        if lastValidationLoss != None and validationLoss.item() > lastValidationLoss :
            converged = True
            pass
        else:
            lastValidationLoss = validationLoss.item()

    endTime = time.time()

    print("Runtime: %s" % (endTime - startTime))

    ##
    # Evaluate the Model
    ##

    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    model.train(mode=False)

    yTestPredicted = None
    yTestActual = None

    for (batchXTensor, batchYTensor) in testDataSetEvaluation:
        # Do the forward pass
        yTestPredicted = model(batchXTensor)
        yTestActual = batchYTensor


    testAccuracy = EvaluateBinaryClassification.Accuracy(yTestActual, [ 1 if pred.item() > 0.5 else 0 for pred in yTestPredicted ])
    print("Accuracy simple:", testAccuracy, ErrorBounds.GetAccuracyBounds(testAccuracy, len(yTestPredicted), 0.50))

# Evaluations:

# lr = 1.0

## I start with a high learning rate of 1.0 to get a sense of how much accuracy hidden nodes provide. 

## 1. 5
# Train is 3556 samples, 48.9033 percent opened.
# Validate is 445 samples, 52.5843 percent opened.
# Test is 445 samples 47.4157 percent opened
# Device is: cpu
# Runtime: 300.21399998664856
# Accuracy simple: 0.8089887640449438 (0.7965035641663075, 0.8214739639235801)

## Evaluation : The model is biased, and starts overfitting after about 7 epochs.
## Next step is to increase the power of the model. I will experiment with the number of nodes in the hidden layer. 

## 2. 50
# Train is 3556 samples, 48.9033 percent opened.
# Validate is 445 samples, 52.5843 percent opened.
# Test is 445 samples 47.4157 percent opened
# Device is: cpu
# Runtime: 145.58951592445374
# Accuracy simple: 0.7797752808988764 (0.7666135635049042, 0.7929369982928486)

## Evaluation : The model is biased, and starts overfitting after about 3 epochs.
## Increasing the number of hidden nodes did not help very much and infact reduced the accuracy
## Next step is go back to a lower the number of hidden nodes. Next step is to model the problem structure better. We will add in the first convulation layer.

## 3. CL1 5

## 4. 10 x 10