kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module2/Assignment1/"

import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import multiprocessing.dummy as mp
import numpy as np
from itertools import repeat
from functools import cmp_to_key
import time
import logging
import time


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


(xRaw, yRaw) = AdultDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

xValidateRaw = np.array(xValidateRaw)
yValidate = np.array(yValidate)
xTestRaw = np.array(xTestRaw)
yTest = np.array(yTest)


print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

for i in range(featurizer.GetFeatureCount()):
    print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

for i in range(10):
    print("%d - " % (yTrain[i]), xTrain[i])

############################

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel

model = MostCommonClassModel.MostCommonClassModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)
validateAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, yValidatePredicted)
errorBounds = ErrorBounds.GetAccuracyBounds(validateAccuracy, len(yValidate), 0.95)

print()
print("### 'Most Common Class' model validate set accuracy: %.4f (95%% %.4f - %.4f)" % (validateAccuracy, errorBounds[0], errorBounds[1]))


errorBounds = ErrorBounds.GetAccuracyBounds(validateAccuracy, len(yValidate), 0.50)

print()
print("### 'Most Common Class' model validate set accuracy: %.4f (50%% %.4f - %.4f)" % (validateAccuracy, errorBounds[0], errorBounds[1]))

#############################



import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting


bestParameters = dict()

bestParameters['stepSize'] = 1.0
bestParameters['convergence'] = 0.001

sweeps = dict()

sweeps['stepSizeSweep'] = [5.0, 3.0, 1.0, .01]
sweeps['convergenceSweep'] = [0.01, 0.005, 0.0001, 0.00005]

confidenceIntervals = [.5, .8, .9, .95, .99]
improvement_accuracies = []
improvement_errorBars = []
improvement_series = ['improvement in accuracy with 50% confidence interval']
improvement_xValues = [0, 1, 2]
min_accuracy_improvement = 1


# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
    global f
    pointsToEvaluate = 100
    thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPR_dict = dict()
    FNR_dict = dict()

    FPRs = []
    FNRs = []

    def f(threshold):
        try:
            FPR = EvaluateBinaryClassification.FalsePositiveRate(yValidate, model.predict(xValidate, classificationThreshold=threshold))
            FNR = EvaluateBinaryClassification.FalseNegativeRate(yValidate, model.predict(xValidate, classificationThreshold=threshold))
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


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    global evaluateFold

    startTime = time.time()
    
    # HERE upgrade this to use crossvalidation

    numberOfFolds = 1


    def evaluateFold(foldId):
        logging.info("evaluating fold %s", foldId)
        numberOfCorrect = 0
        (xTrainRawK, yTrainK, xEvaluateRawK, yEvaluateK) = CrossValidation.CrossValidation(xTrainRaw, yTrain, numberOfFolds, foldId)

        featurizer = AdultFeaturize.AdultFeaturize()
        featurizer.CreateFeatureSet(xTrainRawK, yTrainK, useCategoricalFeatures = True, useNumericFeatures = False)

        xTrainK      = np.asarray(featurizer.Featurize(xTrainRawK))
        xEvaluateK   = np.asarray(featurizer.Featurize(xEvaluateRawK))
        yTrainK      = np.asarray(yTrainK)

        model = LogisticRegression.LogisticRegression()
        model.fit(xTrainK, yTrainK,convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'], verbose=True)
    
        yPredictK = model.predict(xEvaluateK)

        for i in range(len(yPredictK)):
            if yEvaluateK[i] == yPredictK[i]:
                numberOfCorrect += 1
        return numberOfCorrect

    with mp.Pool() as p:
        numberOfCorrects = p.map(evaluateFold, range(numberOfFolds))
    p.close()
    p.join()

    # numberOfCorrects = []
    # for foldId in range(numberOfFolds):
    #     numberOfCorrects.append(evaluateFold(foldId))


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

evaluationRunSpecifications = []

def plotParameterVs50Accuracy(evaluations, sweepName):


    chartTitle = "Plot - Accuracy vs {}".format(sweepName)
    xAxisTitle = sweepName
    yAxisTitle = 'accuracy'
    yAxisTitleRuntime = 'runtime (seconds)'
    fileName = "Plot-Sweep-{}".format(sweepName)
    fileNameRuntime = "Plot-Sweep-{}-runtime".format(sweepName)
    chartTitleRuntime = "Plot - Runtime vs {}".format(sweepName)

    runtime = []
    accuracies = []
    errorBars = []
    series = []
    xValues = []
    min_lowerBound = 1

    min_runtime = 99999999999999
    for confidence in [.5]:
        runtime_s = []
        accuracies_s = []
        errorBars_s = []
        series.append("{} {}".format(sweepName, "{} Accuracy").format(confidence))
        for evaluation in evaluations:
            accuracies_s.append(evaluation['accuracy'])
            errorBars_s.append((evaluation['accuracy'] - evaluation['confidenceIntervals'][confidence]['lowerBound']))
            xValues.append(evaluation[evaluation['optimizing']])
            runtime_s.append(evaluation['runtime'])
            if (evaluation['confidenceIntervals'][confidence]['lowerBound'] < min_lowerBound) :
                min_lowerBound = evaluation['confidenceIntervals'][confidence]['lowerBound']
            if (evaluation['runtime'] < min_runtime) :
                min_runtime = evaluation['runtime']
        runtime.append(runtime_s)
        accuracies.append(accuracies_s)
        errorBars.append(errorBars_s)

    Charting.PlotSeriesWithErrorBars(accuracies, errorBars, series, xValues, chartTitle=chartTitle, xAxisTitle=xAxisTitle, yAxisTitle=yAxisTitle, yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName=fileName)
    Charting.PlotSeries(runtime, series, xValues, chartTitle=chartTitleRuntime, xAxisTitle=xAxisTitle, yAxisTitle=yAxisTitleRuntime, yBotLimit=min_runtime - 20, outputDirectory=kOutputDirectory, fileName=fileNameRuntime)

# returns -1 if a's accuracy is less than b's accuracy, 0 if a's accuracy =  b's accuracy, and 1 if a's accuracy > b's accuracy.
def comparator(a, b):

    if (a['confidenceIntervals'][.5]['upperBound'] <= b['confidenceIntervals'][.5]['lowerBound']):
        return -1
    elif (a['confidenceIntervals'][.5]['lowerBound'] >= b['confidenceIntervals'][.5]['upperBound']):
        return 1
    else:
        if(a['runtime'] > b['runtime']):
            return -1
        elif (a['runtime'] == b['runtime']):
            return 0
        else:
            return 1


def updateBestParameters(evaluations, parameter):
    evaluations = sorted(evaluations, key=lambda a: a[parameter])
    bestEvaluation = evaluations[-1]

    for evaluation in evaluations:
        logging.info(evaluation)

    for evaluation in evaluations:
        if (comparator(evaluation, bestEvaluation) > 0):
            bestEvaluation = evaluation


    # bestParameters[parameter] = bestEvaluation[parameter]

    bestParameters['stepSize'] = bestEvaluation['stepSize']
    bestParameters['convergence']  = bestEvaluation['convergence']

    bestParameters['condifenceInterval'] = bestEvaluation['accuracy'] - bestEvaluation['confidenceIntervals'][.5]['lowerBound']
    bestParameters['accuracy'] = bestEvaluation['accuracy']
    logging.info(bestParameters)

    return evaluations


def updateAccuracyImprovements(accuracy):
    global min_accuracy_improvement

    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidateRaw), .5)

    errorBar = accuracy - lowerBound

    if (min_accuracy_improvement > accuracy - errorBar):
        min_accuracy_improvement = accuracy - errorBar

    improvement_accuracies.append(accuracy)
    improvement_errorBars.append(errorBar)

def plotAccuracyImprovements():
    chartTitle = "Plot - Validation Accuracy Improvement vs number of sweeps"
    xAxisTitle = "Number of optimization sweeps"
    yAxisTitle = 'Validation accuracy'
    fileName = "Plot-AccuracyImprovement"

    Charting.PlotSeriesWithErrorBars([improvement_accuracies], [improvement_errorBars], improvement_series, improvement_xValues, chartTitle=chartTitle, xAxisTitle=xAxisTitle, yAxisTitle=yAxisTitle, yBotLimit=min_accuracy_improvement - 0.01, outputDirectory=kOutputDirectory, fileName=fileName)

def plotAccuracyImprovements():
    chartTitle = "Plot - Validation Accuracy Improvement vs number of sweeps"
    xAxisTitle = "Number of optimization sweeps"
    yAxisTitle = 'Validation accuracy'
    fileName = "Plot-AccuracyImprovement"

    Charting.PlotSeriesWithErrorBars([improvement_accuracies], [improvement_errorBars], improvement_series, improvement_xValues, chartTitle=chartTitle, xAxisTitle=xAxisTitle, yAxisTitle=yAxisTitle, yBotLimit=min_accuracy_improvement - 0.01, outputDirectory=kOutputDirectory, fileName=fileName)

def evaluateOnValidationSet():
    convergence = bestParameters['convergence']
    stepSize = bestParameters['stepSize']
    featurizer = AdultFeaturize.AdultFeaturize()
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

    xTrainEvaluation      = np.asarray(featurizer.Featurize(xTrainRaw))
    xValidateEvaluation   = np.asarray(featurizer.Featurize(xValidateRaw))

    model = LogisticRegression.LogisticRegression()
    model.fit(xTrainEvaluation,yTrain,convergence=convergence, stepSize=stepSize)

    model.visualize()
    yPredict = model.predict(xValidateEvaluation)
    EvaluateBinaryClassification.ExecuteAll(yValidate, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yValidate, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidateRaw), .5)
    print("validation set accuracy lower bound: {}".format(lowerBound))
    print("validation set accuracy upper bound: {}".format(upperBound))
    updateAccuracyImprovements(accuracy)

    return (model, featurizer)

def evaluateOnTestSet(model, featurizer):
    xTestEvaluation   = np.asarray(featurizer.Featurize(xTestRaw))
    yPredict = model.predict(xTestEvaluation)
    EvaluateBinaryClassification.ExecuteAll(yTest, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yTest, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTestRaw), .5)
    print("test set accuracy lower bound: {}".format(lowerBound))
    print("test set accuracy upper bound: {}".format(upperBound))
    

def compareInitialVsBest(model, featurizer):
    testLosses = []
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    convergence = 0.005
    stepSize = 1.0

    featurizer = AdultFeaturize.AdultFeaturize()
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

    initialModelFeaturizer = AdultFeaturize.AdultFeaturize()
    initialModelFeaturizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

    xTrainEvaluation      = np.asarray(initialModelFeaturizer.Featurize(xTrainRaw))
    xTestEvaluation   = np.asarray(initialModelFeaturizer.Featurize(xTestRaw))

    initialModel = LogisticRegression.LogisticRegression()
    initialModel.fit(xTrainEvaluation,yTrain,convergence=convergence, stepSize=stepSize)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(initialModel, xTestEvaluation, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('Initial model')
    print("Rate {}".format(list(zip(thresholds, modelFPRs, modelFNRs))))

    xTestEvaluation   = np.asarray(featurizer.Featurize(xTestRaw))
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xTestEvaluation, yTest)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('Tuned model')

    print("Rate {}".format(list(zip(thresholds, modelFPRs, modelFNRs))))

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="7_Plot-SMSSpamROCs_Optimized")

from joblib import Parallel, delayed

(model, featurizer) = evaluateOnValidationSet()
evaluateOnTestSet(model, featurizer)


evaluationRunSpecifications = []
for convergence in sweeps['convergenceSweep']:

    runSpecification = {}
    runSpecification['optimizing'] = 'convergence'
    runSpecification['stepSize'] = bestParameters['stepSize']
    runSpecification['convergence'] = convergence
    evaluationRunSpecifications.append(runSpecification)

# evaluations = [ ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications ]
evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

logging.info('swept convergence output')
evaluations = updateBestParameters(evaluations, 'convergence')
plotParameterVs50Accuracy(evaluations, 'convergence')
(model, featurizer) = evaluateOnValidationSet()

evaluationRunSpecifications = []

for stepSize in sweeps['stepSizeSweep']:

    runSpecification = {}
    runSpecification['optimizing'] = 'stepSize'
    runSpecification['stepSize'] = stepSize
    runSpecification['convergence'] = bestParameters['convergence']
    evaluationRunSpecifications.append(runSpecification)

evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

logging.info('swept stepsize output')
evaluations = updateBestParameters(evaluations, 'stepSize')
plotParameterVs50Accuracy(evaluations, 'stepSize')
(model, featurizer) = evaluateOnValidationSet()

plotAccuracyImprovements()
evaluateOnTestSet(model, featurizer)
compareInitialVsBest(model, featurizer)