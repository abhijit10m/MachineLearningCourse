kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module2/Assignment5/"

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Learners.AdaBoost as AdaBoost
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize

import numpy as np
import multiprocessing.dummy as mp
import logging
from joblib import Parallel, delayed


sweeps = dict()

sweeps['maxDepth'] = [1,5,10,12,15,18,20,30,35,40]
sweeps['rounds'] = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
sweeps['stepSize'] = [4.0, 3.0, 1.0, 0.8, 0.1, 0.01, 0.001]

# sweeps['maxDepth'] = [12]
# sweeps['rounds'] = [10]
# sweeps['stepSize'] = [0.001]

(xRaw, yRaw) = AdultDataset.LoadRawData()

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

xValidateRaw = np.array(xValidateRaw)
yValidate = np.array(yValidate)
print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True, normalize=True)

xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
xTest   = np.asarray(featurizer.Featurize(xTestRaw))

for i in range(featurizer.GetFeatureCount()):
    print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

for i in range(10):
    print("%d - " % (yTrain[i]), xTrain[i])

def plotValidationVsTrainingSetAccuracy(models, sweepName):
    yValues = []
    errorBars = []
    seriesName = ["Training set accuracy","Validation set accuracy"]
    xValues = []
    sweeps[sweepName]

    yValues_train = []
    errorBars_train = []
    yValues_validate = []
    errorBars_validate = []
    min_lowerBound = 1.0
    counter = 0
    for (sweep, model) in models:
        
        # Training Set Accuracy
        yPredict = model.predict(xTrain)
        EvaluateBinaryClassification.ExecuteAll(yTrain, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTrain, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTrain), .5)
        yValues_train.append(accuracy)
        errorBars_train.append(accuracy - lowerBound)

        # Validation Set Accuracy
        yPredict = model.predict(xValidate)
        EvaluateBinaryClassification.ExecuteAll(yValidate, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yValidate, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidate), .5)
        yValues_validate.append(accuracy)
        errorBars_validate.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        xValues.append(sweeps[sweepName][counter])
        counter += 1
        print("yValues_train %", yValues_train)
        print("yValues_validate %", yValues_validate)
        print("errorBars_train %", errorBars_train)
        print("errorBars_validate %", errorBars_validate)
        print("seriesName %", seriesName)
        print("xValues %", xValues)
        Charting.PlotSeriesWithErrorBars([yValues_train, yValues_validate], [errorBars_train, errorBars_validate], seriesName, xValues, chartTitle="Validation Set vs Training Set accuracy", xAxisTitle=sweepName, yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="TestSetAccuracyVsNumberOfRounds_{}".format(sweepName))



def plotValidationVsTestingSetAccuracy(models, sweepName):
    yValues = []
    errorBars = []
    seriesName = ["Training set accuracy","Testing set accuracy"]
    xValues = []
    sweeps[sweepName]

    yValues_train = []
    errorBars_train = []
    yValues_test = []
    errorBars_test = []
    min_lowerBound = 1.0
    counter = 0
    for (sweep, model) in models:
        
        # Training Set Accuracy
        yPredict = model.predict(xTrain)
        EvaluateBinaryClassification.ExecuteAll(yTrain, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTrain, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTrain), .5)
        yValues_train.append(accuracy)
        errorBars_train.append(accuracy - lowerBound)

        # Validation Set Accuracy
        yPredict = model.predict(xTest)
        EvaluateBinaryClassification.ExecuteAll(yTest, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTest, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTest), .5)
        yValues_test.append(accuracy)
        errorBars_test.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        xValues.append(sweeps[sweepName][counter])
        counter += 1
        print("yValues_train %", yValues_train)
        print("yValues_test %", yValues_test)
        print("errorBars_train %", errorBars_train)
        print("errorBars_test %", errorBars_test)
        print("seriesName %", seriesName)
        print("xValues %", xValues)
        Charting.PlotSeriesWithErrorBars([yValues_train, yValues_test], [errorBars_train, errorBars_test], seriesName, xValues, chartTitle="Validation Set vs Test Set accuracy", xAxisTitle=sweepName, yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="TestSetAccuracyVsNumberOfRounds_{}".format(sweepName))


# # stepSize in logistic regression
# convergence = 0.0001
# models = []


# def evaluateLogisticRegression(stepSize):
#     model = LogisticRegression.LogisticRegression()
#     model.fit(xTrain, np.asarray(yTrain), convergence=convergence, stepSize=stepSize, verbose=True)
#     return (stepSize, model)

# models = Parallel(n_jobs=6)(delayed(evaluateLogisticRegression)(stepSize) for stepSize in sweeps['stepSize'])
# models = sorted(models, key = lambda x: x[0])
# plotValidationVsTrainingSetAccuracy(models, "stepSize")

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True, normalize=False)

xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))

## rounds hyperparamter in BoostedTrees
models = []
maxDepth = 1
adaBoost = AdaBoost.AdaBoost(sweeps['rounds'][-1], DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=maxDepth)
adaBoostModel = adaBoost.adaBoost(xTrain.tolist(), yTrain)


for round in sweeps['rounds']:
    models.append((round, adaBoostModel.getModelWithRounds(round)))

models = sorted(models, key = lambda x: x[0])
plotValidationVsTrainingSetAccuracy(models, "rounds")
plotValidationVsTestingSetAccuracy(models, "rounds")

# # maxDepth for decision trees
# models = []
# def evaluateDecisionTree(maxDepth):
#     model = DecisionTreeWeighted.DecisionTreeWeighted()
#     model.fit(xTrain.tolist(), yTrain,maxDepth=maxDepth)
#     return (maxDepth, model)

# models = Parallel(n_jobs=6)(delayed(evaluateDecisionTree)(maxDepth) for maxDepth in sweeps['maxDepth'])

# models = sorted(models, key = lambda x: x[0])
# plotValidationVsTrainingSetAccuracy(models, "maxDepth")





# # stepSize in logistic regression
# convergence = 0.0001
# models = []

# def evaluateLogisticRegression(stepSize):
#     model = LogisticRegression.LogisticRegression()
#     model.fit(xTrain, np.asarray(yTrain), convergence=convergence, stepSize=stepSize, verbose=True)
#     return (stepSize, model)

# models = Parallel(n_jobs=6)(delayed(evaluateLogisticRegression)(stepSize) for stepSize in sweeps['stepSize'])
# models = sorted(models, key = lambda x: x[0])

# plotValidationVsTestingSetAccuracy(models, "stepSize")


# featurizer = AdultFeaturize.AdultFeaturize()
# featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True, normalize=False)

# xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
# xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))

# ## rounds hyperparamter in BoostedTrees
# models = []
# maxDepth = 12
# rounds = 10
# adaBoost = AdaBoost.AdaBoost(rounds, DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=maxDepth)
# adaBoostModel = adaBoost.adaBoost(xTrain.tolist(), yTrain)

# models.append((rounds, adaBoostModel.getModelWithRounds(rounds)))

# models = sorted(models, key = lambda x: x[0])
# # plotValidationVsTrainingSetAccuracy(models, "rounds")
# plotValidationVsTestingSetAccuracy(models, "rounds")



# ## maxDepth for decision trees

# model = DecisionTreeWeighted.DecisionTreeWeighted()
# model.fit(xTrain.tolist(), yTrain,maxDepth=12)

# models = [(12, model)]

# # plotValidationVsTrainingSetAccuracy(models, "maxDepth")
# plotValidationVsTestingSetAccuracy(models, "maxDepth")