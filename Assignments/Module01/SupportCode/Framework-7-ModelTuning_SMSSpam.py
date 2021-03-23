import multiprocessing.dummy as mp
import numpy as np
from itertools import repeat
from functools import cmp_to_key
kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/ParameterSweep/"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

xValidateRaw = np.array(xValidateRaw)
yValidate = np.array(yValidate)
xTestRaw = np.array(xTestRaw)
yTest = np.array(yTest)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation
import logging

import time

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

bestParameters = dict()

bestParameters['stepSize'] = 1.0
bestParameters['convergence'] = 0.005
bestParameters['numFrequentWords'] = 0
bestParameters['numMutualInformationWords'] = 20

sweeps = dict()
sweeps['numMutualInformationWordsSweep'] = [20, 100, 150, 300, 1000, 1100]
sweeps['numFrequentWordsSweep'] = [0, 100, 200, 300, 350, 400]
sweeps['stepSizeSweep'] = [15.0, 12.0, 10.0, 5.0, 3.0, 1.0]
sweeps['convergenceSweep'] = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]


confidenceIntervals = [.5, .8, .9, .95, .99]



improvement_accuracies = []
improvement_errorBars = []
improvement_series = ['improvement in accuracy with 50% confidence interval']
improvement_xValues = [0, 1, 2, 3, 4]
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

        for threshold in thresholds:
            FPRs.append(FPR_dict[threshold])
            FNRs.append(FNR_dict[threshold])
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting



## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    global evaluateFold

    startTime = time.time()
    
    # HERE upgrade this to use crossvalidation

    numberOfFolds = 5


    def evaluateFold(foldId):
        logging.info("evaluating fold %s", foldId)
        numberOfCorrect = 0
        (xTrainRawK, yTrainK, xEvaluateRawK, yEvaluateK) = CrossValidation.CrossValidation(xTrainRaw, yTrain, numberOfFolds, foldId)

        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xTrainRawK, yTrainK, numFrequentWords = runSpecification['numFrequentWords'], numMutualInformationWords = runSpecification['numMutualInformationWords'])

        xTrainK      = np.asarray(featurizer.Featurize(xTrainRawK))
        xEvaluateK   = np.asarray(featurizer.Featurize(xEvaluateRawK))
        yTrainK      = np.asarray(yTrainK)

        frequentModel = LogisticRegression.LogisticRegression()
        frequentModel.fit(xTrainK, yTrainK,convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'], verbose=True)
    
        yPredictK = frequentModel.predict(xEvaluateK)

        for i in range(len(yPredictK)):
            if yEvaluateK[i] == yPredictK[i]:
                numberOfCorrect += 1
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

evaluationRunSpecifications = []

def plotParameterVs50Accuracy(evaluations, sweepName):


    chartTitle = "Plot - Accuracy vs {}".format(sweepName, sweepName)
    xAxisTitle = sweepName
    yAxisTitle = 'accuracy'
    yAxisTitleRuntime = 'runtime (seconds)'
    fileName = "Plot-Sweep-{}".format(sweepName)
    fileNameRuntime = "Plot-Sweep-{}-runtime".format(sweepName)
    chartTitleRuntime = "Plot - Runtime vs {}".format(sweepName, sweepName)

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
    bestParameters['numFrequentWords']  = bestEvaluation['numFrequentWords']
    bestParameters['numMutualInformationWords']  = bestEvaluation['numMutualInformationWords']

    bestParameters['condifenceInterval'] = bestEvaluation['accuracy'] - bestEvaluation['confidenceIntervals'][.5]['lowerBound']
    bestParameters['accuracy'] = bestEvaluation['accuracy']
    # logging.info('updated bestParameters')
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

def evaluateOnValidationSet():
    numFrequentWords = bestParameters['numFrequentWords']
    numMutualInformationWords = bestParameters['numMutualInformationWords']
    convergence = bestParameters['convergence']
    stepSize = bestParameters['stepSize']
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = numFrequentWords, numMutualInformationWords = numMutualInformationWords)

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
    
    initialParameters = dict()
    initialParameters['stepSize'] = 1.0
    initialParameters['convergence'] = 0.005
    initialParameters['numFrequentWords'] = 0
    initialParameters['numMutualInformationWords'] = 20

    numFrequentWords = initialParameters['numFrequentWords']
    numMutualInformationWords = initialParameters['numMutualInformationWords']
    convergence = initialParameters['convergence']
    stepSize = initialParameters['stepSize']


    initialModelFeaturizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
    initialModelFeaturizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = numFrequentWords, numMutualInformationWords = numMutualInformationWords)

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

# (model, featurizer) = evaluateOnValidationSet()
# evaluateOnTestSet(model, featurizer)

# evaluationRunSpecifications = []
# for numMutualInformationWords in sweeps['numMutualInformationWordsSweep']:

#     runSpecification = {}
#     runSpecification['optimizing'] = 'numMutualInformationWords'
#     runSpecification['numMutualInformationWords'] = numMutualInformationWords
#     runSpecification['stepSize'] = bestParameters['stepSize']
#     runSpecification['convergence'] = bestParameters['convergence']
#     runSpecification['numFrequentWords'] = bestParameters['numFrequentWords']
#     evaluationRunSpecifications.append(runSpecification)


# ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
# evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)


# evaluations = [
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9148401826484018, 'confidenceIntervals': {0.5: {'lowerBound': 0.9120144717667388, 'upperBound': 0.9176658935300649}, 0.8: {'lowerBound': 0.9094418096207469, 'upperBound': 0.9202385556760567}, 0.9: {'lowerBound': 0.907923517206719, 'upperBound': 0.9217568480900846}, 0.95: {'lowerBound': 0.9065739239498053, 'upperBound': 0.9231064413469984}, 0.99: {'lowerBound': 0.903959087014535, 'upperBound': 0.9257212782822687}}, 'runtime': 223.8293797969818},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9351598173515981, 'confidenceIntervals': {0.5: {'lowerBound': 0.9326669292717521, 'upperBound': 0.9376527054314442}, 0.8: {'lowerBound': 0.930397284900549, 'upperBound': 0.9399223498026473}, 0.9: {'lowerBound': 0.9290578226486914, 'upperBound': 0.9412618120545049}, 0.95: {'lowerBound': 0.9278671895359291, 'upperBound': 0.9424524451672672}, 0.99: {'lowerBound': 0.9255603378799522, 'upperBound': 0.9447592968232441}}, 'runtime': 251.13365507125854},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9406392694063926, 'confidenceIntervals': {0.5: {'lowerBound': 0.9382470615136134, 'upperBound': 0.9430314772991719}, 0.8: {'lowerBound': 0.9360690811933217, 'upperBound': 0.9452094576194636}, 0.9: {'lowerBound': 0.9347837157583955, 'upperBound': 0.9464948230543898}, 0.95: {'lowerBound': 0.9336411687051277, 'upperBound': 0.9476373701076576}, 0.99: {'lowerBound': 0.9314274837894215, 'upperBound': 0.9498510550233638}}, 'runtime': 256.1351399421692},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 300, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9429223744292238, 'confidenceIntervals': {0.5: {'lowerBound': 0.9405737765352381, 'upperBound': 0.9452709723232094}, 0.8: {'lowerBound': 0.9384355008407138, 'upperBound': 0.9474092480177337}, 0.9: {'lowerBound': 0.9371735676439453, 'upperBound': 0.9486711812145022}, 0.95: {'lowerBound': 0.9360518492468178, 'upperBound': 0.9497928996116297}, 0.99: {'lowerBound': 0.9338785198523832, 'upperBound': 0.9519662290060643}}, 'runtime': 268.47381019592285},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 306.4628839492798},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 1100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.95, 'confidenceIntervals': {0.5: {'lowerBound': 0.9477935984584493, 'upperBound': 0.9522064015415506}, 0.8: {'lowerBound': 0.9457847851146495, 'upperBound': 0.9542152148853504}, 0.9: {'lowerBound': 0.9445992559281446, 'upperBound': 0.9554007440718553}, 0.95: {'lowerBound': 0.943545452206807, 'upperBound': 0.9564545477931929}, 0.99: {'lowerBound': 0.9415037074967154, 'upperBound': 0.9584962925032845}}, 'runtime': 312.37823486328125}
# ]


# logging.info('swept numMutualInformationWords output')
# evaluations = updateBestParameters(evaluations, 'numMutualInformationWords')
# plotParameterVs50Accuracy(evaluations, 'numMutualInformationWords')
# (model, featurizer) = evaluateOnValidationSet()

# evaluationRunSpecifications = []

# for numFrequentWords in sweeps['numFrequentWordsSweep']:

#     runSpecification = {}
#     runSpecification['optimizing'] = 'numFrequentWords'
#     runSpecification['numMutualInformationWords'] = bestParameters['numMutualInformationWords']
#     runSpecification['stepSize'] = bestParameters['stepSize']
#     runSpecification['convergence'] = bestParameters['convergence']
#     runSpecification['numFrequentWords'] = numFrequentWords
#     evaluationRunSpecifications.append(runSpecification)

# evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

# evaluations = [
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 328.7204649448395},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9511415525114155, 'confidenceIntervals': {0.5: {'lowerBound': 0.9489591735996623, 'upperBound': 0.9533239314231687}, 0.8: {'lowerBound': 0.9469722316053795, 'upperBound': 0.9553108734174516}, 0.9: {'lowerBound': 0.9457996101005568, 'upperBound': 0.9564834949222742}, 0.95: {'lowerBound': 0.9447572798740478, 'upperBound': 0.9575258251487833}, 0.99: {'lowerBound': 0.9427377650601866, 'upperBound': 0.9595453399626445}}, 'runtime': 333.5397298336029},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 200, 'accuracy': 0.9504566210045662, 'confidenceIntervals': {0.5: {'lowerBound': 0.9482597896962844, 'upperBound': 0.9526534523128479}, 0.8: {'lowerBound': 0.9462596895499384, 'upperBound': 0.9546535524591939}, 0.9: {'lowerBound': 0.9450793025783244, 'upperBound': 0.955833939430808}, 0.95: {'lowerBound': 0.9440300697146675, 'upperBound': 0.9568831722944648}, 0.99: {'lowerBound': 0.9419971810413322, 'upperBound': 0.9589160609678001}}, 'runtime': 339.1100468635559},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 300, 'accuracy': 0.9511415525114155, 'confidenceIntervals': {0.5: {'lowerBound': 0.9489591735996623, 'upperBound': 0.9533239314231687}, 0.8: {'lowerBound': 0.9469722316053795, 'upperBound': 0.9553108734174516}, 0.9: {'lowerBound': 0.9457996101005568, 'upperBound': 0.9564834949222742}, 0.95: {'lowerBound': 0.9447572798740478, 'upperBound': 0.9575258251487833}, 0.99: {'lowerBound': 0.9427377650601866, 'upperBound': 0.9595453399626445}}, 'runtime': 344.13117694854736},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 350, 'accuracy': 0.9509132420091324, 'confidenceIntervals': {0.5: {'lowerBound': 0.9487260325785487, 'upperBound': 0.9531004514397161}, 0.8: {'lowerBound': 0.9467346926492112, 'upperBound': 0.9550917913690536}, 0.9: {'lowerBound': 0.9455594756417334, 'upperBound': 0.9562670083765314}, 0.95: {'lowerBound': 0.9445148383017531, 'upperBound': 0.9573116457165117}, 0.99: {'lowerBound': 0.9424908534555412, 'upperBound': 0.9593356305627236}}, 'runtime': 346.9228699207306},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 400, 'accuracy': 0.9518264840182649, 'confidenceIntervals': {0.5: {'lowerBound': 0.9496586760307565, 'upperBound': 0.9539942920057732}, 0.8: {'lowerBound': 0.947685000101831, 'upperBound': 0.9559679679346987}, 0.9: {'lowerBound': 0.946520207750334, 'upperBound': 0.9571327602861958}, 0.95: {'lowerBound': 0.9454848367712255, 'upperBound': 0.9581681312653042}, 0.99: {'lowerBound': 0.9434788054992028, 'upperBound': 0.9601741625373269}}, 'runtime': 346.3782262802124}
# ]

# logging.info('swept numFrequentWords output')
# evaluations = updateBestParameters(evaluations, 'numFrequentWords')
# plotParameterVs50Accuracy(evaluations, 'numFrequentWords')
# (model, featurizer) = evaluateOnValidationSet()


# evaluationRunSpecifications = []
# for convergence in sweeps['convergenceSweep']:

#     runSpecification = {}
#     runSpecification['optimizing'] = 'convergence'
#     runSpecification['numMutualInformationWords'] = bestParameters['numMutualInformationWords']
#     runSpecification['stepSize'] = bestParameters['stepSize']
#     runSpecification['convergence'] = convergence
#     runSpecification['numFrequentWords'] = bestParameters['numFrequentWords']
#     evaluationRunSpecifications.append(runSpecification)

# evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

# evaluations = [
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 5e-05, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 3779.344512939453},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9751141552511415, 'confidenceIntervals': {0.5: {'lowerBound': 0.9735371190177567, 'upperBound': 0.9766911914845263}, 0.8: {'lowerBound': 0.9721013099097496, 'upperBound': 0.9781270005925334}, 0.9: {'lowerBound': 0.9712539471574833, 'upperBound': 0.9789743633447998}, 0.95: {'lowerBound': 0.9705007358221353, 'upperBound': 0.9797275746801478}, 0.99: {'lowerBound': 0.9690413888598985, 'upperBound': 0.9811869216423845}}, 'runtime': 2497.6194009780884},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0005, 'numFrequentWords': 0, 'accuracy': 0.967351598173516, 'confidenceIntervals': {0.5: {'lowerBound': 0.9655524750406977, 'upperBound': 0.9691507213063343}, 0.8: {'lowerBound': 0.9639144674123109, 'upperBound': 0.9707887289347211}, 0.9: {'lowerBound': 0.9629477743857219, 'upperBound': 0.9717554219613102}, 0.95: {'lowerBound': 0.9620884916954207, 'upperBound': 0.9726147046516114}, 0.99: {'lowerBound': 0.9604236314829618, 'upperBound': 0.9742795648640702}}, 'runtime': 972.5662608146667},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.001, 'numFrequentWords': 0, 'accuracy': 0.9634703196347032, 'confidenceIntervals': {0.5: {'lowerBound': 0.9615710796113233, 'upperBound': 0.9653695596580831}, 0.8: {'lowerBound': 0.9598419207840668, 'upperBound': 0.9670987184853396}, 0.9: {'lowerBound': 0.9588214336073254, 'upperBound': 0.968119205662081}, 0.95: {'lowerBound': 0.9579143338946663, 'upperBound': 0.9690263053747401}, 0.99: {'lowerBound': 0.9561568282013894, 'upperBound': 0.970783811068017}}, 'runtime': 678.3698103427887},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 309.4766981601715},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.01, 'numFrequentWords': 0, 'accuracy': 0.9388127853881278, 'confidenceIntervals': {0.5: {'lowerBound': 0.9363864122633678, 'upperBound': 0.9412391585128879}, 0.8: {'lowerBound': 0.9341773262841087, 'upperBound': 0.943448244492147}, 0.9: {'lowerBound': 0.9328736034111033, 'upperBound': 0.9447519673651524}, 0.95: {'lowerBound': 0.9317147386350985, 'upperBound': 0.9459108321411572}, 0.99: {'lowerBound': 0.9294694381315892, 'upperBound': 0.9481561326446665}}, 'runtime': 239.19428300857544}
# ]

# logging.info('swept convergence output')
# evaluations = updateBestParameters(evaluations, 'convergence')
# plotParameterVs50Accuracy(evaluations, 'convergence')
# (model, featurizer) = evaluateOnValidationSet()


# evaluationRunSpecifications = []

# for stepSize in sweeps['stepSizeSweep']:

#     runSpecification = {}
#     runSpecification['optimizing'] = 'stepSize'
#     runSpecification['numMutualInformationWords'] = bestParameters['numMutualInformationWords']
#     runSpecification['stepSize'] = stepSize
#     runSpecification['convergence'] = bestParameters['convergence']
#     runSpecification['numFrequentWords'] = bestParameters['numFrequentWords']
#     evaluationRunSpecifications.append(runSpecification)

# evaluations = Parallel(n_jobs=6)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

evaluations = [
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9751141552511415, 'confidenceIntervals': {0.5: {'lowerBound': 0.9735371190177567, 'upperBound': 0.9766911914845263}, 0.8: {'lowerBound': 0.9721013099097496, 'upperBound': 0.9781270005925334}, 0.9: {'lowerBound': 0.9712539471574833, 'upperBound': 0.9789743633447998}, 0.95: {'lowerBound': 0.9705007358221353, 'upperBound': 0.9797275746801478}, 0.99: {'lowerBound': 0.9690413888598985, 'upperBound': 0.9811869216423845}}, 'runtime': 3263.5032498836517},
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 3.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9762557077625571, 'confidenceIntervals': {0.5: {'lowerBound': 0.9747143652578463, 'upperBound': 0.9777970502672679}, 0.8: {'lowerBound': 0.9733110534251991, 'upperBound': 0.9792003620999151}, 0.9: {'lowerBound': 0.9724828693928171, 'upperBound': 0.9800285461322971}, 0.95: {'lowerBound': 0.9717467058084777, 'upperBound': 0.9807647097166365}, 0.99: {'lowerBound': 0.9703203888638199, 'upperBound': 0.9821910266612943}}, 'runtime': 2824.1224870681763},
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 5.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9769406392694064, 'confidenceIntervals': {0.5: {'lowerBound': 0.9754211575935792, 'upperBound': 0.9784601209452337}, 0.8: {'lowerBound': 0.9740377489036469, 'upperBound': 0.979843529635166}, 0.9: {'lowerBound': 0.973221310988277, 'upperBound': 0.9806599675505359}, 0.95: {'lowerBound': 0.9724955883968371, 'upperBound': 0.9813856901419757}, 0.99: {'lowerBound': 0.9710895008759223, 'upperBound': 0.9827917776628906}}, 'runtime': 2598.1711769104004},
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 10.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 2176.9077258110046},
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 12.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 2043.715533733368},
    {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 15.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9764840182648402, 'confidenceIntervals': {0.5: {'lowerBound': 0.9749499246072737, 'upperBound': 0.9780181119224067}, 0.8: {'lowerBound': 0.9735532124712802, 'upperBound': 0.9794148240584002}, 0.9: {'lowerBound': 0.9727289233418416, 'upperBound': 0.9802391131878389}, 0.95: {'lowerBound': 0.9719962218934516, 'upperBound': 0.9809718146362288}, 0.99: {'lowerBound': 0.970576612837196, 'upperBound': 0.9823914236924844}}, 'runtime': 1913.1294980049133}
]

logging.info('swept stepsize output')
evaluations = updateBestParameters(evaluations, 'stepSize')
# plotParameterVs50Accuracy(evaluations, 'stepSize')
(model, featurizer) = evaluateOnValidationSet()

# plotAccuracyImprovements()
evaluateOnTestSet(model, featurizer)
compareInitialVsBest(model, featurizer)

# Good luck!


# sweeps['numMutualInformationWordsSweep'] = [20, 25, 50, 75, 100, 125]
# sweeps['numFrequentWordsSweep'] = [0, 25, 50, 75, 100, 125]
# sweeps['stepSizeSweep'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
# sweeps['convergenceSweep'] = [0.005, 0.0001, 0.0005, 0.0001, 0.00005, 0.00001] 
# evaluations = [
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.952054794520548, 'confidenceIntervals': {0.5: {'lowerBound': 0.9498918702607474, 'upperBound': 0.9542177187803486}, 0.8: {'lowerBound': 0.9479226407107796, 'upperBound': 0.9561869483303164}, 0.9: {'lowerBound': 0.9467604724517823, 'upperBound': 0.9573491165893137}, 0.95: {'lowerBound': 0.9457274339993402, 'upperBound': 0.9583821550417557}, 0.99: {'lowerBound': 0.9437259219977336, 'upperBound': 0.9603836670433623}}, 'runtime': 234.74140310287476},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 25, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.9513698630136986, 'confidenceIntervals': {0.5: {'lowerBound': 0.9491923277898456, 'upperBound': 0.9535473982375516}, 0.8: {'lowerBound': 0.9472097957203675, 'upperBound': 0.9555299303070298}, 0.9: {'lowerBound': 0.9460397767941181, 'upperBound': 0.9566999492332792}, 0.95: {'lowerBound': 0.9449997599707853, 'upperBound': 0.9577399660566119}, 0.99: {'lowerBound': 0.942984727375578, 'upperBound': 0.9597549986518192}}, 'runtime': 235.06654405593872},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 50, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.9579908675799087, 'confidenceIntervals': {0.5: {'lowerBound': 0.9559599594028086, 'upperBound': 0.9600217757570088}, 0.8: {'lowerBound': 0.9541109235997772, 'upperBound': 0.9618708115600402}, 0.9: {'lowerBound': 0.9530196893553652, 'upperBound': 0.9629620458044522}, 0.95: {'lowerBound': 0.9520497033603323, 'upperBound': 0.9639320317994851}, 0.99: {'lowerBound': 0.9501703554949562, 'upperBound': 0.9658113796648612}}, 'runtime': 236.80981302261353},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 75, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.9584474885844749, 'confidenceIntervals': {0.5: {'lowerBound': 0.956427166793764, 'upperBound': 0.9604678103751858}, 0.8: {'lowerBound': 0.9545877693425199, 'upperBound': 0.96230720782643}, 0.9: {'lowerBound': 0.95350222330572, 'upperBound': 0.9633927538632299}, 0.95: {'lowerBound': 0.9525372934952312, 'upperBound': 0.9643576836737187}, 0.99: {'lowerBound': 0.9506677419874092, 'upperBound': 0.9662272351815406}}, 'runtime': 238.10546827316284},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.9584474885844749, 'confidenceIntervals': {0.5: {'lowerBound': 0.956427166793764, 'upperBound': 0.9604678103751858}, 0.8: {'lowerBound': 0.9545877693425199, 'upperBound': 0.96230720782643}, 0.9: {'lowerBound': 0.95350222330572, 'upperBound': 0.9633927538632299}, 0.95: {'lowerBound': 0.9525372934952312, 'upperBound': 0.9643576836737187}, 0.99: {'lowerBound': 0.9506677419874092, 'upperBound': 0.9662272351815406}}, 'runtime': 240.3135859966278},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 125, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.960730593607306, 'confidenceIntervals': {0.5: {'lowerBound': 0.9587642214058798, 'upperBound': 0.9626969658087321}, 0.8: {'lowerBound': 0.9569739422374173, 'upperBound': 0.9644872449771946}, 0.9: {'lowerBound': 0.9559173840396361, 'upperBound': 0.9655438031749758}, 0.95: {'lowerBound': 0.9549782211971639, 'upperBound': 0.966482966017448}, 0.99: {'lowerBound': 0.9531585931898741, 'upperBound': 0.9683025940247378}}, 'runtime': 240.36837482452393}
# ]


# sweeps['numMutualInformationWordsSweep'] = [20, 25, 50, 75, 100, 125]
# sweeps['numFrequentWordsSweep'] = [0, 25, 50, 75, 100, 125]
# sweeps['stepSizeSweep'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8]
# sweeps['convergenceSweep'] = [0.005, 0.0001, 0.0005, 0.0001, 0.00005, 0.00001] 

# evaluations = [
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 0.8, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9107305936073059, 'confidenceIntervals': {0.5: {'lowerBound': 0.9078440107123135, 'upperBound': 0.9136171765022983}, 0.8: {'lowerBound': 0.9052159277780667, 'upperBound': 0.9162452594365451}, 0.9: {'lowerBound': 0.9036649280135931, 'upperBound': 0.9177962592010187}, 0.95: {'lowerBound': 0.9022862615562833, 'upperBound': 0.9191749256583285}, 0.99: {'lowerBound': 0.8996150952952455, 'upperBound': 0.9218460919193663}}, 'runtime': 357.0928211212158},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9148401826484018, 'confidenceIntervals': {0.5: {'lowerBound': 0.9120144717667388, 'upperBound': 0.9176658935300649}, 0.8: {'lowerBound': 0.9094418096207469, 'upperBound': 0.9202385556760567}, 0.9: {'lowerBound': 0.907923517206719, 'upperBound': 0.9217568480900846}, 0.95: {'lowerBound': 0.9065739239498053, 'upperBound': 0.9231064413469984}, 0.99: {'lowerBound': 0.903959087014535, 'upperBound': 0.9257212782822687}}, 'runtime': 345.79311776161194},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 2.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9223744292237442, 'confidenceIntervals': {0.5: {'lowerBound': 0.9196655242036582, 'upperBound': 0.9250833342438303}, 0.8: {'lowerBound': 0.9171992076928337, 'upperBound': 0.9275496507546548}, 0.9: {'lowerBound': 0.9157436766372651, 'upperBound': 0.9290051818102234}, 0.95: {'lowerBound': 0.9144498712545374, 'upperBound': 0.9302989871929511}, 0.99: {'lowerBound': 0.9119431233255026, 'upperBound': 0.9328057351219858}}, 'runtime': 314.4094271659851},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 3.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9283105022831051, 'confidenceIntervals': {0.5: {'lowerBound': 0.925698869279984, 'upperBound': 0.9309221352862261}, 0.8: {'lowerBound': 0.9233211138592321, 'upperBound': 0.933299890706978}, 0.9: {'lowerBound': 0.9219178483650178, 'upperBound': 0.9347031562011924}, 0.95: {'lowerBound': 0.9206705012590496, 'upperBound': 0.9359505033071606}, 0.99: {'lowerBound': 0.9182537662412361, 'upperBound': 0.938367238324974}}, 'runtime': 299.3233850002289},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 4.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9319634703196347, 'confidenceIntervals': {0.5: {'lowerBound': 0.9294142447311239, 'upperBound': 0.9345126959081455}, 0.8: {'lowerBound': 0.9270933080012856, 'upperBound': 0.9368336326379837}, 0.9: {'lowerBound': 0.92572357484925, 'upperBound': 0.9382033657900193}, 0.95: {'lowerBound': 0.9245060342696628, 'upperBound': 0.9394209063696065}, 0.99: {'lowerBound': 0.9221470493967125, 'upperBound': 0.9417798912425568}}, 'runtime': 297.5915949344635},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9328767123287671, 'confidenceIntervals': {0.5: {'lowerBound': 0.9303434131443767, 'upperBound': 0.9354100115131575}, 0.8: {'lowerBound': 0.9280369765735139, 'upperBound': 0.9377164480840203}, 0.9: {'lowerBound': 0.926675800892349, 'upperBound': 0.9390776237651852}, 0.95: {'lowerBound': 0.9254658669535357, 'upperBound': 0.9402875577039985}, 0.99: {'lowerBound': 0.923121619947085, 'upperBound': 0.9426318047104493}}, 'runtime': 289.7871458530426}
# ]


# sweeps['numMutualInformationWordsSweep'] = [20, 25, 50, 75, 100, 125]
# sweeps['numFrequentWordsSweep'] = [0, 25, 50, 75, 100, 125]
# sweeps['stepSizeSweep'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
# sweeps['convergenceSweep'] = [0.005, 0.0001, 0.0005, 0.0001, 0.00005, 0.00001] 
# evaluations = [
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9328767123287671, 'confidenceIntervals': {0.5: {'lowerBound': 0.9303434131443767, 'upperBound': 0.9354100115131575}, 0.8: {'lowerBound': 0.9280369765735139, 'upperBound': 0.9377164480840203}, 0.9: {'lowerBound': 0.926675800892349, 'upperBound': 0.9390776237651852}, 0.95: {'lowerBound': 0.9254658669535357, 'upperBound': 0.9402875577039985}, 0.99: {'lowerBound': 0.923121619947085, 'upperBound': 0.9426318047104493}}, 'runtime': 206.06153988838196},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 25, 'accuracy': 0.936986301369863, 'confidenceIntervals': {0.5: {'lowerBound': 0.9345263764388921, 'upperBound': 0.939446226300834}, 0.8: {'lowerBound': 0.9322867432927844, 'upperBound': 0.9416858594469417}, 0.9: {'lowerBound': 0.930964992583606, 'upperBound': 0.9430076101561201}, 0.95: {'lowerBound': 0.9297901030643363, 'upperBound': 0.9441824996753898}, 0.99: {'lowerBound': 0.9275137546207514, 'upperBound': 0.9464588481189747}}, 'runtime': 207.91268491744995},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.941324200913242, 'confidenceIntervals': {0.5: {'lowerBound': 0.9389449685097292, 'upperBound': 0.9437034333167548}, 0.8: {'lowerBound': 0.9367788016945907, 'upperBound': 0.9458696001318933}, 0.9: {'lowerBound': 0.935500408164345, 'upperBound': 0.947147993662139}, 0.95: {'lowerBound': 0.9343640583596822, 'upperBound': 0.9482843434668018}, 0.99: {'lowerBound': 0.9321623806131479, 'upperBound': 0.950486021213336}}, 'runtime': 212.06537795066833},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 75, 'accuracy': 0.9472602739726027, 'confidenceIntervals': {0.5: {'lowerBound': 0.9449974991160862, 'upperBound': 0.9495230488291192}, 0.8: {'lowerBound': 0.9429373608138847, 'upperBound': 0.9515831871313207}, 0.9: {'lowerBound': 0.9417215414879953, 'upperBound': 0.9527990064572102}, 0.95: {'lowerBound': 0.9406408131983157, 'upperBound': 0.9538797347468897}, 0.99: {'lowerBound': 0.9385469021370617, 'upperBound': 0.9559736458081437}}, 'runtime': 222.3693389892578},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9472602739726027, 'confidenceIntervals': {0.5: {'lowerBound': 0.9449974991160862, 'upperBound': 0.9495230488291192}, 0.8: {'lowerBound': 0.9429373608138847, 'upperBound': 0.9515831871313207}, 0.9: {'lowerBound': 0.9417215414879953, 'upperBound': 0.9527990064572102}, 0.95: {'lowerBound': 0.9406408131983157, 'upperBound': 0.9538797347468897}, 0.99: {'lowerBound': 0.9385469021370617, 'upperBound': 0.9559736458081437}}, 'runtime': 222.92858386039734},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'accuracy': 0.952054794520548, 'confidenceIntervals': {0.5: {'lowerBound': 0.9498918702607474, 'upperBound': 0.9542177187803486}, 0.8: {'lowerBound': 0.9479226407107796, 'upperBound': 0.9561869483303164}, 0.9: {'lowerBound': 0.9467604724517823, 'upperBound': 0.9573491165893137}, 0.95: {'lowerBound': 0.9457274339993402, 'upperBound': 0.9583821550417557}, 0.99: {'lowerBound': 0.9437259219977336, 'upperBound': 0.9603836670433623}}, 'runtime': 224.5879499912262}
# ]

# sweeps['numMutualInformationWordsSweep'] = [20, 25, 50, 75, 100, 125]
# sweeps['numFrequentWordsSweep'] = [0, 25, 50, 75, 100, 125]
# sweeps['stepSizeSweep'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
# sweeps['convergenceSweep'] = [0.005, 0.0001, 0.0005, 0.0001, 0.00005, 0.00001] 
# evaluations = [
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 1e-05, 'numFrequentWords': 0, 'accuracy': 0.9397260273972603, 'confidenceIntervals': {0.5: {'lowerBound': 0.9373166585902861, 'upperBound': 0.9421353962042345}, 0.8: {'lowerBound': 0.9351230541540857, 'upperBound': 0.9443290006404349}, 0.9: {'lowerBound': 0.9338284679294429, 'upperBound': 0.9456235868650777}, 0.95: {'lowerBound': 0.9326777246186493, 'upperBound': 0.9467743301758713}, 0.99: {'lowerBound': 0.9304481594539866, 'upperBound': 0.949003895340534}}, 'runtime': 1241.3822658061981},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 5e-05, 'numFrequentWords': 0, 'accuracy': 0.9388127853881278, 'confidenceIntervals': {0.5: {'lowerBound': 0.9363864122633678, 'upperBound': 0.9412391585128879}, 0.8: {'lowerBound': 0.9341773262841087, 'upperBound': 0.943448244492147}, 0.9: {'lowerBound': 0.9328736034111033, 'upperBound': 0.9447519673651524}, 0.95: {'lowerBound': 0.9317147386350985, 'upperBound': 0.9459108321411572}, 0.99: {'lowerBound': 0.9294694381315892, 'upperBound': 0.9481561326446665}}, 'runtime': 859.7161159515381},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9397260273972603, 'confidenceIntervals': {0.5: {'lowerBound': 0.9373166585902861, 'upperBound': 0.9421353962042345}, 0.8: {'lowerBound': 0.9351230541540857, 'upperBound': 0.9443290006404349}, 0.9: {'lowerBound': 0.9338284679294429, 'upperBound': 0.9456235868650777}, 0.95: {'lowerBound': 0.9326777246186493, 'upperBound': 0.9467743301758713}, 0.99: {'lowerBound': 0.9304481594539866, 'upperBound': 0.949003895340534}}, 'runtime': 740.3294789791107},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.0005, 'numFrequentWords': 0, 'accuracy': 0.9406392694063926, 'confidenceIntervals': {0.5: {'lowerBound': 0.9382470615136134, 'upperBound': 0.9430314772991719}, 0.8: {'lowerBound': 0.9360690811933217, 'upperBound': 0.9452094576194636}, 0.9: {'lowerBound': 0.9347837157583955, 'upperBound': 0.9464948230543898}, 0.95: {'lowerBound': 0.9336411687051277, 'upperBound': 0.9476373701076576}, 0.99: {'lowerBound': 0.9314274837894215, 'upperBound': 0.9498510550233638}}, 'runtime': 467.1736857891083},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.001, 'numFrequentWords': 0, 'accuracy': 0.9367579908675799, 'confidenceIntervals': {0.5: {'lowerBound': 0.934293913839339, 'upperBound': 0.9392220678958209}, 0.8: {'lowerBound': 0.9320505004255675, 'upperBound': 0.9414654813095924}, 0.9: {'lowerBound': 0.9307265187387516, 'upperBound': 0.9427894629964083}, 0.95: {'lowerBound': 0.9295496461282484, 'upperBound': 0.9439663356069115}, 0.99: {'lowerBound': 0.9272694554453986, 'upperBound': 0.9462465262897612}}, 'runtime': 374.79100704193115},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 20, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9328767123287671, 'confidenceIntervals': {0.5: {'lowerBound': 0.9303434131443767, 'upperBound': 0.9354100115131575}, 0.8: {'lowerBound': 0.9280369765735139, 'upperBound': 0.9377164480840203}, 0.9: {'lowerBound': 0.926675800892349, 'upperBound': 0.9390776237651852}, 0.95: {'lowerBound': 0.9254658669535357, 'upperBound': 0.9402875577039985}, 0.99: {'lowerBound': 0.923121619947085, 'upperBound': 0.9426318047104493}}, 'runtime': 236.35044693946838}
# ]

# Best evaluation: 
# Results 1:
# sweeps['numMutualInformationWordsSweep'] = [20, 25, 50, 75, 100, 125]
# sweeps['numFrequentWordsSweep'] = [0, 25, 50, 75, 100, 125]
# sweeps['stepSizeSweep'] = [5.0, 4.0, 3.0, 2.0, 1.0, 0.8]
# sweeps['convergenceSweep'] = [0.005, 0.0001, 0.0005, 0.0001, 0.00005, 0.00001] 
#                  15:14:34: {'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 125, 'numMutualInformationWords': 50}

#Results 2:
# evaluations = [
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9148401826484018, 'confidenceIntervals': {0.5: {'lowerBound': 0.9120144717667388, 'upperBound': 0.9176658935300649}, 0.8: {'lowerBound': 0.9094418096207469, 'upperBound': 0.9202385556760567}, 0.9: {'lowerBound': 0.907923517206719, 'upperBound': 0.9217568480900846}, 0.95: {'lowerBound': 0.9065739239498053, 'upperBound': 0.9231064413469984}, 0.99: {'lowerBound': 0.903959087014535, 'upperBound': 0.9257212782822687}}, 'runtime': 38.433939933776855},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 50, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9273972602739726, 'confidenceIntervals': {0.5: {'lowerBound': 0.9247703384021916, 'upperBound': 0.9300241821457537}, 0.8: {'lowerBound': 0.9223786632651969, 'upperBound': 0.9324158572827483}, 0.9: {'lowerBound': 0.9209671828564787, 'upperBound': 0.9338273376914665}, 0.95: {'lowerBound': 0.9197125336042847, 'upperBound': 0.9350819869436605}, 0.99: {'lowerBound': 0.917281650678159, 'upperBound': 0.9375128698697862}}, 'runtime': 41.97480916976929},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.934931506849315, 'confidenceIntervals': {0.5: {'lowerBound': 0.9324345386001922, 'upperBound': 0.9374284750984379}, 0.8: {'lowerBound': 0.9301611794480058, 'upperBound': 0.9397018342506243}, 0.9: {'lowerBound': 0.9288195248663875, 'upperBound': 0.9410434888322425}, 0.95: {'lowerBound': 0.9276269430160602, 'upperBound': 0.9422360706825699}, 0.99: {'lowerBound': 0.9253163156810511, 'upperBound': 0.944546698017579}}, 'runtime': 42.37801909446716},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9367579908675799, 'confidenceIntervals': {0.5: {'lowerBound': 0.934293913839339, 'upperBound': 0.9392220678958209}, 0.8: {'lowerBound': 0.9320505004255675, 'upperBound': 0.9414654813095924}, 0.9: {'lowerBound': 0.9307265187387516, 'upperBound': 0.9427894629964083}, 0.95: {'lowerBound': 0.9295496461282484, 'upperBound': 0.9439663356069115}, 0.99: {'lowerBound': 0.9272694554453986, 'upperBound': 0.9462465262897612}}, 'runtime': 44.60141086578369},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 200, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.939041095890411, 'confidenceIntervals': {0.5: {'lowerBound': 0.9366189593450679, 'upperBound': 0.941463232435754}, 0.8: {'lowerBound': 0.9344137305500541, 'upperBound': 0.9436684612307679}, 0.9: {'lowerBound': 0.9331122840480787, 'upperBound': 0.9449699077327433}, 0.95: {'lowerBound': 0.9319554427129895, 'upperBound': 0.9461267490678325}, 0.99: {'lowerBound': 0.929714062626254, 'upperBound': 0.9483681291545679}}, 'runtime': 44.926666021347046},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 250, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9397260273972603, 'confidenceIntervals': {0.5: {'lowerBound': 0.9373166585902861, 'upperBound': 0.9421353962042345}, 0.8: {'lowerBound': 0.9351230541540857, 'upperBound': 0.9443290006404349}, 0.9: {'lowerBound': 0.9338284679294429, 'upperBound': 0.9456235868650777}, 0.95: {'lowerBound': 0.9326777246186493, 'upperBound': 0.9467743301758713}, 0.99: {'lowerBound': 0.9304481594539866, 'upperBound': 0.949003895340534}}, 'runtime': 45.60758590698242}
# ]

# evaluations = [
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.934931506849315, 'confidenceIntervals': {0.5: {'lowerBound': 0.9324345386001922, 'upperBound': 0.9374284750984379}, 0.8: {'lowerBound': 0.9301611794480058, 'upperBound': 0.9397018342506243}, 0.9: {'lowerBound': 0.9288195248663875, 'upperBound': 0.9410434888322425}, 0.95: {'lowerBound': 0.9276269430160602, 'upperBound': 0.9422360706825699}, 0.99: {'lowerBound': 0.9253163156810511, 'upperBound': 0.944546698017579}}, 'runtime': 37.58616614341736},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 25, 'accuracy': 0.936986301369863, 'confidenceIntervals': {0.5: {'lowerBound': 0.9345263764388921, 'upperBound': 0.939446226300834}, 0.8: {'lowerBound': 0.9322867432927844, 'upperBound': 0.9416858594469417}, 0.9: {'lowerBound': 0.930964992583606, 'upperBound': 0.9430076101561201}, 0.95: {'lowerBound': 0.9297901030643363, 'upperBound': 0.9441824996753898}, 0.99: {'lowerBound': 0.9275137546207514, 'upperBound': 0.9464588481189747}}, 'runtime': 39.67248296737671},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9408675799086758, 'confidenceIntervals': {0.5: {'lowerBound': 0.938479687107606, 'upperBound': 0.9432554727097455}, 0.8: {'lowerBound': 0.9363056354529006, 'upperBound': 0.9454295243644509}, 0.9: {'lowerBound': 0.9350225885747139, 'upperBound': 0.9467125712426376}, 0.95: {'lowerBound': 0.9338821024607701, 'upperBound': 0.9478530573565814}, 0.99: {'lowerBound': 0.9316724106150039, 'upperBound': 0.9500627492023476}}, 'runtime': 39.70148992538452},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9429223744292238, 'confidenceIntervals': {0.5: {'lowerBound': 0.9405737765352381, 'upperBound': 0.9452709723232094}, 0.8: {'lowerBound': 0.9384355008407138, 'upperBound': 0.9474092480177337}, 0.9: {'lowerBound': 0.9371735676439453, 'upperBound': 0.9486711812145022}, 0.95: {'lowerBound': 0.9360518492468178, 'upperBound': 0.9497928996116297}, 0.99: {'lowerBound': 0.9338785198523832, 'upperBound': 0.9519662290060643}}, 'runtime': 40.11390280723572},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 150, 'accuracy': 0.9438356164383561, 'confidenceIntervals': {0.5: {'lowerBound': 0.9415047551628476, 'upperBound': 0.9461664777138646}, 0.8: {'lowerBound': 0.9393826277329069, 'upperBound': 0.9482886051438053}, 0.9: {'lowerBound': 0.9381302246594994, 'upperBound': 0.9495410082172129}, 0.95: {'lowerBound': 0.9370169774831371, 'upperBound': 0.9506542553935752}, 0.99: {'lowerBound': 0.9348600610789352, 'upperBound': 0.9528111717977771}}, 'runtime': 40.629920959472656},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 200, 'accuracy': 0.9454337899543379, 'confidenceIntervals': {0.5: {'lowerBound': 0.9431343863770889, 'upperBound': 0.9477331935315869}, 0.8: {'lowerBound': 0.9410408995381009, 'upperBound': 0.9498266803705749}, 0.9: {'lowerBound': 0.9398053991085342, 'upperBound': 0.9510621808001416}, 0.95: {'lowerBound': 0.938707176504475, 'upperBound': 0.9521604034042008}, 0.99: {'lowerBound': 0.9365793702091102, 'upperBound': 0.9542882096995656}}, 'runtime': 41.256587743759155}
# ]


# evaluations = [
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 0.8, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9356164383561644, 'confidenceIntervals': {0.5: {'lowerBound': 0.9331317371652896, 'upperBound': 0.9381011395470391}, 0.8: {'lowerBound': 0.9308695465288215, 'upperBound': 0.9403633301835073}, 0.9: {'lowerBound': 0.9295344832023813, 'upperBound': 0.9416983935099474}, 0.95: {'lowerBound': 0.9283477602455457, 'upperBound': 0.9428851164667831}, 0.99: {'lowerBound': 0.9260484845166764, 'upperBound': 0.9451843921956523}}, 'runtime': 38.32133483886719},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9408675799086758, 'confidenceIntervals': {0.5: {'lowerBound': 0.938479687107606, 'upperBound': 0.9432554727097455}, 0.8: {'lowerBound': 0.9363056354529006, 'upperBound': 0.9454295243644509}, 0.9: {'lowerBound': 0.9350225885747139, 'upperBound': 0.9467125712426376}, 0.95: {'lowerBound': 0.9338821024607701, 'upperBound': 0.9478530573565814}, 0.99: {'lowerBound': 0.9316724106150039, 'upperBound': 0.9500627492023476}}, 'runtime': 37.852355003356934},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 2.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9495433789954338, 'confidenceIntervals': {0.5: {'lowerBound': 0.9473274581959092, 'upperBound': 0.9517592997949584}, 0.8: {'lowerBound': 0.9453099780649988, 'upperBound': 0.9537767799258688}, 0.9: {'lowerBound': 0.9441193340533139, 'upperBound': 0.9549674239375537}, 0.95: {'lowerBound': 0.9430609838207051, 'upperBound': 0.9560257741701625}, 0.99: {'lowerBound': 0.9410104302450256, 'upperBound': 0.9580763277458421}}, 'runtime': 34.80730175971985},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 3.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9536529680365297, 'confidenceIntervals': {0.5: {'lowerBound': 0.9515246138916503, 'upperBound': 0.955781322181409}, 0.8: {'lowerBound': 0.9495868586254168, 'upperBound': 0.9577190774476425}, 0.9: {'lowerBound': 0.9484432653535413, 'upperBound': 0.958862670719518}, 0.95: {'lowerBound': 0.9474267380007632, 'upperBound': 0.9598791980722962}, 0.99: {'lowerBound': 0.9454572162547553, 'upperBound': 0.961848719818304}}, 'runtime': 33.376203775405884},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 4.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9559360730593607, 'confidenceIntervals': {0.5: {'lowerBound': 0.9538583207585831, 'upperBound': 0.9580138253601382}, 0.8: {'lowerBound': 0.9519666358280244, 'upperBound': 0.9599055102906969}, 0.9: {'lowerBound': 0.9508502316067111, 'upperBound': 0.9610219145120102}, 0.95: {'lowerBound': 0.9498578722988771, 'upperBound': 0.9620142738198443}, 0.99: {'lowerBound': 0.9479351761399486, 'upperBound': 0.9639369699787728}}, 'runtime': 32.877235889434814},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9573059360730594, 'confidenceIntervals': {0.5: {'lowerBound': 0.9552592705610173, 'upperBound': 0.9593526015851014}, 0.8: {'lowerBound': 0.9533958885276655, 'upperBound': 0.9612159836184532}, 0.9: {'lowerBound': 0.9522961876555236, 'upperBound': 0.9623156844905951}, 0.95: {'lowerBound': 0.9513186757691752, 'upperBound': 0.9632931963769436}, 0.99: {'lowerBound': 0.949424746489375, 'upperBound': 0.9651871256567437}}, 'runtime': 30.74498200416565}
# ]

# evaluations = [
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 5e-05, 'numFrequentWords': 50, 'accuracy': 0.967351598173516, 'confidenceIntervals': {0.5: {'lowerBound': 0.9655524750406977, 'upperBound': 0.9691507213063343}, 0.8: {'lowerBound': 0.9639144674123109, 'upperBound': 0.9707887289347211}, 0.9: {'lowerBound': 0.9629477743857219, 'upperBound': 0.9717554219613102}, 0.95: {'lowerBound': 0.9620884916954207, 'upperBound': 0.9726147046516114}, 0.99: {'lowerBound': 0.9604236314829618, 'upperBound': 0.9742795648640702}}, 'runtime': 198.98004007339478},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.0001, 'numFrequentWords': 50, 'accuracy': 0.967351598173516, 'confidenceIntervals': {0.5: {'lowerBound': 0.9655524750406977, 'upperBound': 0.9691507213063343}, 0.8: {'lowerBound': 0.9639144674123109, 'upperBound': 0.9707887289347211}, 0.9: {'lowerBound': 0.9629477743857219, 'upperBound': 0.9717554219613102}, 0.95: {'lowerBound': 0.9620884916954207, 'upperBound': 0.9726147046516114}, 0.99: {'lowerBound': 0.9604236314829618, 'upperBound': 0.9742795648640702}}, 'runtime': 142.90507197380066},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.0005, 'numFrequentWords': 50, 'accuracy': 0.9655251141552511, 'confidenceIntervals': {0.5: {'lowerBound': 0.9636780968981769, 'upperBound': 0.9673721314123254}, 0.8: {'lowerBound': 0.9619964841715869, 'upperBound': 0.9690537441389153}, 0.9: {'lowerBound': 0.9610040569886814, 'upperBound': 0.9700461713218208}, 0.95: {'lowerBound': 0.9601218994927653, 'upperBound': 0.9709283288177369}, 0.99: {'lowerBound': 0.958412719344428, 'upperBound': 0.9726375089660743}}, 'runtime': 69.3873999118805},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.001, 'numFrequentWords': 50, 'accuracy': 0.9632420091324201, 'confidenceIntervals': {0.5: {'lowerBound': 0.9613370689726641, 'upperBound': 0.965146949292176}, 0.8: {'lowerBound': 0.9596027204690057, 'upperBound': 0.9668812977958344}, 0.9: {'lowerBound': 0.9585791705324205, 'upperBound': 0.9679048477324197}, 0.95: {'lowerBound': 0.9576693483665669, 'upperBound': 0.9688146698982733}, 0.99: {'lowerBound': 0.9559065679202257, 'upperBound': 0.9705774503446145}}, 'runtime': 54.479729890823364},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9573059360730594, 'confidenceIntervals': {0.5: {'lowerBound': 0.9552592705610173, 'upperBound': 0.9593526015851014}, 0.8: {'lowerBound': 0.9533958885276655, 'upperBound': 0.9612159836184532}, 0.9: {'lowerBound': 0.9522961876555236, 'upperBound': 0.9623156844905951}, 0.95: {'lowerBound': 0.9513186757691752, 'upperBound': 0.9632931963769436}, 0.99: {'lowerBound': 0.949424746489375, 'upperBound': 0.9651871256567437}}, 'runtime': 32.84686017036438},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 100, 'stepSize': 5.0, 'convergence': 0.01, 'numFrequentWords': 50, 'accuracy': 0.9536529680365297, 'confidenceIntervals': {0.5: {'lowerBound': 0.9515246138916503, 'upperBound': 0.955781322181409}, 0.8: {'lowerBound': 0.9495868586254168, 'upperBound': 0.9577190774476425}, 0.9: {'lowerBound': 0.9484432653535413, 'upperBound': 0.958862670719518}, 0.95: {'lowerBound': 0.9474267380007632, 'upperBound': 0.9598791980722962}, 0.99: {'lowerBound': 0.9454572162547553, 'upperBound': 0.961848719818304}}, 'runtime': 28.65312623977661}
# ]

# Results 3
# evaluations = [
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9148401826484018, 'confidenceIntervals': {0.5: {'lowerBound': 0.9120144717667388, 'upperBound': 0.9176658935300649}, 0.8: {'lowerBound': 0.9094418096207469, 'upperBound': 0.9202385556760567}, 0.9: {'lowerBound': 0.907923517206719, 'upperBound': 0.9217568480900846}, 0.95: {'lowerBound': 0.9065739239498053, 'upperBound': 0.9231064413469984}, 0.99: {'lowerBound': 0.903959087014535, 'upperBound': 0.9257212782822687}}, 'runtime': 226.41916418075562},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 50, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9305936073059361, 'confidenceIntervals': {0.5: {'lowerBound': 0.9280207392410942, 'upperBound': 0.933166475370778}, 0.8: {'lowerBound': 0.9256782772716112, 'upperBound': 0.935508937340261}, 0.9: {'lowerBound': 0.9242958406994574, 'upperBound': 0.9368913739124148}, 0.95: {'lowerBound': 0.9230670081908762, 'upperBound': 0.938120206420996}, 0.99: {'lowerBound': 0.9206861452055, 'upperBound': 0.9405010694063722}}, 'runtime': 237.68171906471252},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9351598173515981, 'confidenceIntervals': {0.5: {'lowerBound': 0.9326669292717521, 'upperBound': 0.9376527054314442}, 0.8: {'lowerBound': 0.930397284900549, 'upperBound': 0.9399223498026473}, 0.9: {'lowerBound': 0.9290578226486914, 'upperBound': 0.9412618120545049}, 0.95: {'lowerBound': 0.9278671895359291, 'upperBound': 0.9424524451672672}, 0.99: {'lowerBound': 0.9255603378799522, 'upperBound': 0.9447592968232441}}, 'runtime': 249.6294069290161},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9406392694063926, 'confidenceIntervals': {0.5: {'lowerBound': 0.9382470615136134, 'upperBound': 0.9430314772991719}, 0.8: {'lowerBound': 0.9360690811933217, 'upperBound': 0.9452094576194636}, 0.9: {'lowerBound': 0.9347837157583955, 'upperBound': 0.9464948230543898}, 0.95: {'lowerBound': 0.9336411687051277, 'upperBound': 0.9476373701076576}, 0.99: {'lowerBound': 0.9314274837894215, 'upperBound': 0.9498510550233638}}, 'runtime': 252.0937259197235},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 200, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9415525114155251, 'confidenceIntervals': {0.5: {'lowerBound': 0.9391776244280404, 'upperBound': 0.9439273984030098}, 0.8: {'lowerBound': 0.9370154138871961, 'upperBound': 0.9460896089438541}, 0.9: {'lowerBound': 0.9357393552073535, 'upperBound': 0.9473656676236967}, 0.95: {'lowerBound': 0.9346050808252713, 'upperBound': 0.9484999420057789}, 0.99: {'lowerBound': 0.9324074242099869, 'upperBound': 0.9506975986210633}}, 'runtime': 256.43648982048035},
#     {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 250, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9417808219178082, 'confidenceIntervals': {0.5: {'lowerBound': 0.939410290565527, 'upperBound': 0.9441513532700894}, 0.8: {'lowerBound': 0.9372520456030025, 'upperBound': 0.946309598232614}, 0.9: {'lowerBound': 0.9359783272644634, 'upperBound': 0.947583316571153}, 0.95: {'lowerBound': 0.9348461331857619, 'upperBound': 0.9487155106498545}, 0.99: {'lowerBound': 0.9326525071582779, 'upperBound': 0.9509091366773386}}, 'runtime': 260.1558291912079}
# ]

# evaluations = [
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9406392694063926, 'confidenceIntervals': {0.5: {'lowerBound': 0.9382470615136134, 'upperBound': 0.9430314772991719}, 0.8: {'lowerBound': 0.9360690811933217, 'upperBound': 0.9452094576194636}, 0.9: {'lowerBound': 0.9347837157583955, 'upperBound': 0.9464948230543898}, 0.95: {'lowerBound': 0.9336411687051277, 'upperBound': 0.9476373701076576}, 0.99: {'lowerBound': 0.9314274837894215, 'upperBound': 0.9498510550233638}}, 'runtime': 264.79427099227905},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 25, 'accuracy': 0.9429223744292238, 'confidenceIntervals': {0.5: {'lowerBound': 0.9405737765352381, 'upperBound': 0.9452709723232094}, 0.8: {'lowerBound': 0.9384355008407138, 'upperBound': 0.9474092480177337}, 0.9: {'lowerBound': 0.9371735676439453, 'upperBound': 0.9486711812145022}, 0.95: {'lowerBound': 0.9360518492468178, 'upperBound': 0.9497928996116297}, 0.99: {'lowerBound': 0.9338785198523832, 'upperBound': 0.9519662290060643}}, 'runtime': 265.7475531101227},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 50, 'accuracy': 0.9442922374429223, 'confidenceIntervals': {0.5: {'lowerBound': 0.9419703090876046, 'upperBound': 0.9466141657982401}, 0.8: {'lowerBound': 0.9398563146148528, 'upperBound': 0.9487281602709919}, 0.9: {'lowerBound': 0.9386087113194581, 'upperBound': 0.9499757635663866}, 0.95: {'lowerBound': 0.9374997306124407, 'upperBound': 0.951084744273404}, 0.99: {'lowerBound': 0.9353510804925945, 'upperBound': 0.9532333943932502}}, 'runtime': 268.14118790626526},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9484018264840183, 'confidenceIntervals': {0.5: {'lowerBound': 0.9461623262992187, 'upperBound': 0.9506413266688178}, 0.8: {'lowerBound': 0.9441233783697743, 'upperBound': 0.9526802745982622}, 0.9: {'lowerBound': 0.9429200648376432, 'upperBound': 0.9538835881303933}, 0.95: {'lowerBound': 0.9418504528090822, 'upperBound': 0.9549532001589544}, 0.99: {'lowerBound': 0.9397780795037453, 'upperBound': 0.9570255734642913}}, 'runtime': 269.9795331954956},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 150, 'accuracy': 0.9490867579908676, 'confidenceIntervals': {0.5: {'lowerBound': 0.946861368254513, 'upperBound': 0.9513121477272223}, 0.8: {'lowerBound': 0.9448352671512646, 'upperBound': 0.9533382488304706}, 0.9: {'lowerBound': 0.9436395353526263, 'upperBound': 0.9545339806291089}, 0.95: {'lowerBound': 0.9425766626427257, 'upperBound': 0.9555968533390096}, 0.99: {'lowerBound': 0.9405173467672929, 'upperBound': 0.9576561692144423}}, 'runtime': 272.6227810382843},
#     {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 200, 'accuracy': 0.9495433789954338, 'confidenceIntervals': {0.5: {'lowerBound': 0.9473274581959092, 'upperBound': 0.9517592997949584}, 0.8: {'lowerBound': 0.9453099780649988, 'upperBound': 0.9537767799258688}, 0.9: {'lowerBound': 0.9441193340533139, 'upperBound': 0.9549674239375537}, 0.95: {'lowerBound': 0.9430609838207051, 'upperBound': 0.9560257741701625}, 0.99: {'lowerBound': 0.9410104302450256, 'upperBound': 0.9580763277458421}}, 'runtime': 274.89907813072205}
# ]

# evaluations = [
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 0.8, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9445205479452055, 'confidenceIntervals': {0.5: {'lowerBound': 0.9422031024202477, 'upperBound': 0.9468379934701632}, 0.8: {'lowerBound': 0.9400931893303608, 'upperBound': 0.9489479065600501}, 0.9: {'lowerBound': 0.9388479947199357, 'upperBound': 0.9501931011704752}, 0.95: {'lowerBound': 0.9377411550662246, 'upperBound': 0.9512999408241863}, 0.99: {'lowerBound': 0.9355966532371592, 'upperBound': 0.9534444426532517}}, 'runtime': 239.28398203849792},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9484018264840183, 'confidenceIntervals': {0.5: {'lowerBound': 0.9461623262992187, 'upperBound': 0.9506413266688178}, 0.8: {'lowerBound': 0.9441233783697743, 'upperBound': 0.9526802745982622}, 0.9: {'lowerBound': 0.9429200648376432, 'upperBound': 0.9538835881303933}, 0.95: {'lowerBound': 0.9418504528090822, 'upperBound': 0.9549532001589544}, 0.99: {'lowerBound': 0.9397780795037453, 'upperBound': 0.9570255734642913}}, 'runtime': 231.60040402412415},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 2.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9557077625570777, 'confidenceIntervals': {0.5: {'lowerBound': 0.9536248832090981, 'upperBound': 0.9577906419050572}, 0.8: {'lowerBound': 0.9517285303698928, 'upperBound': 0.9596869947442626}, 0.9: {'lowerBound': 0.9506093713172471, 'upperBound': 0.9608061537969083}, 0.95: {'lowerBound': 0.9496145632704509, 'upperBound': 0.9618009618437044}, 0.99: {'lowerBound': 0.9476871226797833, 'upperBound': 0.9637284024343721}}, 'runtime': 211.98637700080872},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 3.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9595890410958904, 'confidenceIntervals': {0.5: {'lowerBound': 0.9575954780937572, 'upperBound': 0.9615826040980235}, 0.8: {'lowerBound': 0.9557804431216659, 'upperBound': 0.9633976390701149}, 0.9: {'lowerBound': 0.9547092749414152, 'upperBound': 0.9644688072503655}, 0.95: {'lowerBound': 0.9537571254478591, 'upperBound': 0.9654209567439217}, 0.99: {'lowerBound': 0.9519123358040941, 'upperBound': 0.9672657463876867}}, 'runtime': 199.908118724823},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 4.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9609589041095891, 'confidenceIntervals': {0.5: {'lowerBound': 0.9589980234862393, 'upperBound': 0.9629197847329388}, 0.8: {'lowerBound': 0.9572127441127418, 'upperBound': 0.9647050641064363}, 0.9: {'lowerBound': 0.9561591366136285, 'upperBound': 0.9657586716055496}, 0.95: {'lowerBound': 0.9552225966144167, 'upperBound': 0.9666952116047615}, 0.99: {'lowerBound': 0.9534080503659438, 'upperBound': 0.9685097578532343}}, 'runtime': 193.37376618385315},
#     {'optimizing': 'stepSize', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9614155251141553, 'confidenceIntervals': {0.5: {'lowerBound': 0.9594656822655451, 'upperBound': 0.9633653679627655}, 0.8: {'lowerBound': 0.9576904522093477, 'upperBound': 0.9651405980189629}, 0.9: {'lowerBound': 0.9566427754548705, 'upperBound': 0.96618827477344}, 0.95: {'lowerBound': 0.9557115072286687, 'upperBound': 0.9671195429996419}, 0.99: {'lowerBound': 0.9539071750404025, 'upperBound': 0.9689238751879081}}, 'runtime': 178.61586499214172}\
# ]
# evaluations = [
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 5e-05, 'numFrequentWords': 100, 'accuracy': 0.973744292237443, 'confidenceIntervals': {0.5: {'lowerBound': 0.9721255709618938, 'upperBound': 0.9753630135129921}, 0.8: {'lowerBound': 0.9706518098005729, 'upperBound': 0.9768367746743131}, 0.9: {'lowerBound': 0.9697820491152032, 'upperBound': 0.9777065353596828}, 0.95: {'lowerBound': 0.9690089285059856, 'upperBound': 0.9784796559689003}, 0.99: {'lowerBound': 0.9675110073256267, 'upperBound': 0.9799775771492593}}, 'runtime': 1505.5170459747314},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.0001, 'numFrequentWords': 100, 'accuracy': 0.9728310502283105, 'confidenceIntervals': {0.5: {'lowerBound': 0.9711851902502409, 'upperBound': 0.9744769102063802}, 0.8: {'lowerBound': 0.9696867207179684, 'upperBound': 0.9759753797386527}, 0.9: {'lowerBound': 0.9688023780431847, 'upperBound': 0.9768597224134363}, 0.95: {'lowerBound': 0.9680162956655991, 'upperBound': 0.9776458047910219}, 0.99: {'lowerBound': 0.9664932610590272, 'upperBound': 0.9791688393975938}}, 'runtime': 1059.7195348739624},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.0005, 'numFrequentWords': 100, 'accuracy': 0.9698630136986301, 'confidenceIntervals': {0.5: {'lowerBound': 0.9681322297591426, 'upperBound': 0.9715937976381177}, 0.8: {'lowerBound': 0.9665564413963255, 'upperBound': 0.9731695860009348}, 0.9: {'lowerBound': 0.9656264679363022, 'upperBound': 0.974099559460958}, 0.95: {'lowerBound': 0.9647998248607261, 'upperBound': 0.9749262025365342}, 0.99: {'lowerBound': 0.9631982039017972, 'upperBound': 0.9765278234954631}}, 'runtime': 471.71690011024475},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.001, 'numFrequentWords': 100, 'accuracy': 0.9675799086757991, 'confidenceIntervals': {0.5: {'lowerBound': 0.9657868756647139, 'upperBound': 0.9693729416868844}, 0.8: {'lowerBound': 0.9641544127740242, 'upperBound': 0.971005404577574}, 0.9: {'lowerBound': 0.96319099205165, 'upperBound': 0.9719688252999482}, 0.95: {'lowerBound': 0.9623346180762063, 'upperBound': 0.9728251992753919}, 0.99: {'lowerBound': 0.9606753934987842, 'upperBound': 0.9744844238528141}}, 'runtime': 342.8280680179596},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9614155251141553, 'confidenceIntervals': {0.5: {'lowerBound': 0.9594656822655451, 'upperBound': 0.9633653679627655}, 0.8: {'lowerBound': 0.9576904522093477, 'upperBound': 0.9651405980189629}, 0.9: {'lowerBound': 0.9566427754548705, 'upperBound': 0.96618827477344}, 0.95: {'lowerBound': 0.9557115072286687, 'upperBound': 0.9671195429996419}, 0.99: {'lowerBound': 0.9539071750404025, 'upperBound': 0.9689238751879081}}, 'runtime': 182.73916697502136},
#     {'optimizing': 'convergence', 'numMutualInformationWords': 150, 'stepSize': 5.0, 'convergence': 0.01, 'numFrequentWords': 100, 'accuracy': 0.9595890410958904, 'confidenceIntervals': {0.5: {'lowerBound': 0.9575954780937572, 'upperBound': 0.9615826040980235}, 0.8: {'lowerBound': 0.9557804431216659, 'upperBound': 0.9633976390701149}, 0.9: {'lowerBound': 0.9547092749414152, 'upperBound': 0.9644688072503655}, 0.95: {'lowerBound': 0.9537571254478591, 'upperBound': 0.9654209567439217}, 0.99: {'lowerBound': 0.9519123358040941, 'upperBound': 0.9672657463876867}}, 'runtime': 161.60815286636353}
# ]

# Results 4
# sweeps['numMutualInformationWordsSweep'] = [20, 100, 150, 300, 1000, 1100]
# sweeps['numFrequentWordsSweep'] = [0, 100, 200, 300, 350, 400]
# sweeps['stepSizeSweep'] = [15.0, 12.0, 10.0, 5.0, 3.0, 1.0]
# sweeps['convergenceSweep'] = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]

# evaluations = [
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 20, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9148401826484018, 'confidenceIntervals': {0.5: {'lowerBound': 0.9120144717667388, 'upperBound': 0.9176658935300649}, 0.8: {'lowerBound': 0.9094418096207469, 'upperBound': 0.9202385556760567}, 0.9: {'lowerBound': 0.907923517206719, 'upperBound': 0.9217568480900846}, 0.95: {'lowerBound': 0.9065739239498053, 'upperBound': 0.9231064413469984}, 0.99: {'lowerBound': 0.903959087014535, 'upperBound': 0.9257212782822687}}, 'runtime': 223.8293797969818},
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9351598173515981, 'confidenceIntervals': {0.5: {'lowerBound': 0.9326669292717521, 'upperBound': 0.9376527054314442}, 0.8: {'lowerBound': 0.930397284900549, 'upperBound': 0.9399223498026473}, 0.9: {'lowerBound': 0.9290578226486914, 'upperBound': 0.9412618120545049}, 0.95: {'lowerBound': 0.9278671895359291, 'upperBound': 0.9424524451672672}, 0.99: {'lowerBound': 0.9255603378799522, 'upperBound': 0.9447592968232441}}, 'runtime': 251.13365507125854},
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 150, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9406392694063926, 'confidenceIntervals': {0.5: {'lowerBound': 0.9382470615136134, 'upperBound': 0.9430314772991719}, 0.8: {'lowerBound': 0.9360690811933217, 'upperBound': 0.9452094576194636}, 0.9: {'lowerBound': 0.9347837157583955, 'upperBound': 0.9464948230543898}, 0.95: {'lowerBound': 0.9336411687051277, 'upperBound': 0.9476373701076576}, 0.99: {'lowerBound': 0.9314274837894215, 'upperBound': 0.9498510550233638}}, 'runtime': 256.1351399421692},
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 300, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9429223744292238, 'confidenceIntervals': {0.5: {'lowerBound': 0.9405737765352381, 'upperBound': 0.9452709723232094}, 0.8: {'lowerBound': 0.9384355008407138, 'upperBound': 0.9474092480177337}, 0.9: {'lowerBound': 0.9371735676439453, 'upperBound': 0.9486711812145022}, 0.95: {'lowerBound': 0.9360518492468178, 'upperBound': 0.9497928996116297}, 0.99: {'lowerBound': 0.9338785198523832, 'upperBound': 0.9519662290060643}}, 'runtime': 268.47381019592285},
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 306.4628839492798},
    # {'optimizing': 'numMutualInformationWords', 'numMutualInformationWords': 1100, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.95, 'confidenceIntervals': {0.5: {'lowerBound': 0.9477935984584493, 'upperBound': 0.9522064015415506}, 0.8: {'lowerBound': 0.9457847851146495, 'upperBound': 0.9542152148853504}, 0.9: {'lowerBound': 0.9445992559281446, 'upperBound': 0.9554007440718553}, 0.95: {'lowerBound': 0.943545452206807, 'upperBound': 0.9564545477931929}, 0.99: {'lowerBound': 0.9415037074967154, 'upperBound': 0.9584962925032845}}, 'runtime': 312.37823486328125}
# ]

# evaluations = [
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 328.7204649448395},
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 100, 'accuracy': 0.9511415525114155, 'confidenceIntervals': {0.5: {'lowerBound': 0.9489591735996623, 'upperBound': 0.9533239314231687}, 0.8: {'lowerBound': 0.9469722316053795, 'upperBound': 0.9553108734174516}, 0.9: {'lowerBound': 0.9457996101005568, 'upperBound': 0.9564834949222742}, 0.95: {'lowerBound': 0.9447572798740478, 'upperBound': 0.9575258251487833}, 0.99: {'lowerBound': 0.9427377650601866, 'upperBound': 0.9595453399626445}}, 'runtime': 333.5397298336029},
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 200, 'accuracy': 0.9504566210045662, 'confidenceIntervals': {0.5: {'lowerBound': 0.9482597896962844, 'upperBound': 0.9526534523128479}, 0.8: {'lowerBound': 0.9462596895499384, 'upperBound': 0.9546535524591939}, 0.9: {'lowerBound': 0.9450793025783244, 'upperBound': 0.955833939430808}, 0.95: {'lowerBound': 0.9440300697146675, 'upperBound': 0.9568831722944648}, 0.99: {'lowerBound': 0.9419971810413322, 'upperBound': 0.9589160609678001}}, 'runtime': 339.1100468635559},
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 300, 'accuracy': 0.9511415525114155, 'confidenceIntervals': {0.5: {'lowerBound': 0.9489591735996623, 'upperBound': 0.9533239314231687}, 0.8: {'lowerBound': 0.9469722316053795, 'upperBound': 0.9553108734174516}, 0.9: {'lowerBound': 0.9457996101005568, 'upperBound': 0.9564834949222742}, 0.95: {'lowerBound': 0.9447572798740478, 'upperBound': 0.9575258251487833}, 0.99: {'lowerBound': 0.9427377650601866, 'upperBound': 0.9595453399626445}}, 'runtime': 344.13117694854736},
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 350, 'accuracy': 0.9509132420091324, 'confidenceIntervals': {0.5: {'lowerBound': 0.9487260325785487, 'upperBound': 0.9531004514397161}, 0.8: {'lowerBound': 0.9467346926492112, 'upperBound': 0.9550917913690536}, 0.9: {'lowerBound': 0.9455594756417334, 'upperBound': 0.9562670083765314}, 0.95: {'lowerBound': 0.9445148383017531, 'upperBound': 0.9573116457165117}, 0.99: {'lowerBound': 0.9424908534555412, 'upperBound': 0.9593356305627236}}, 'runtime': 346.9228699207306},
    # {'optimizing': 'numFrequentWords', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 400, 'accuracy': 0.9518264840182649, 'confidenceIntervals': {0.5: {'lowerBound': 0.9496586760307565, 'upperBound': 0.9539942920057732}, 0.8: {'lowerBound': 0.947685000101831, 'upperBound': 0.9559679679346987}, 0.9: {'lowerBound': 0.946520207750334, 'upperBound': 0.9571327602861958}, 0.95: {'lowerBound': 0.9454848367712255, 'upperBound': 0.9581681312653042}, 0.99: {'lowerBound': 0.9434788054992028, 'upperBound': 0.9601741625373269}}, 'runtime': 346.3782262802124}
# ]

# evaluations = [
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 5e-05, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 3779.344512939453},
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9751141552511415, 'confidenceIntervals': {0.5: {'lowerBound': 0.9735371190177567, 'upperBound': 0.9766911914845263}, 0.8: {'lowerBound': 0.9721013099097496, 'upperBound': 0.9781270005925334}, 0.9: {'lowerBound': 0.9712539471574833, 'upperBound': 0.9789743633447998}, 0.95: {'lowerBound': 0.9705007358221353, 'upperBound': 0.9797275746801478}, 0.99: {'lowerBound': 0.9690413888598985, 'upperBound': 0.9811869216423845}}, 'runtime': 2497.6194009780884},
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0005, 'numFrequentWords': 0, 'accuracy': 0.967351598173516, 'confidenceIntervals': {0.5: {'lowerBound': 0.9655524750406977, 'upperBound': 0.9691507213063343}, 0.8: {'lowerBound': 0.9639144674123109, 'upperBound': 0.9707887289347211}, 0.9: {'lowerBound': 0.9629477743857219, 'upperBound': 0.9717554219613102}, 0.95: {'lowerBound': 0.9620884916954207, 'upperBound': 0.9726147046516114}, 0.99: {'lowerBound': 0.9604236314829618, 'upperBound': 0.9742795648640702}}, 'runtime': 972.5662608146667},
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.001, 'numFrequentWords': 0, 'accuracy': 0.9634703196347032, 'confidenceIntervals': {0.5: {'lowerBound': 0.9615710796113233, 'upperBound': 0.9653695596580831}, 0.8: {'lowerBound': 0.9598419207840668, 'upperBound': 0.9670987184853396}, 0.9: {'lowerBound': 0.9588214336073254, 'upperBound': 0.968119205662081}, 0.95: {'lowerBound': 0.9579143338946663, 'upperBound': 0.9690263053747401}, 0.99: {'lowerBound': 0.9561568282013894, 'upperBound': 0.970783811068017}}, 'runtime': 678.3698103427887},
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.005, 'numFrequentWords': 0, 'accuracy': 0.9474885844748858, 'confidenceIntervals': {0.5: {'lowerBound': 0.9452304406296623, 'upperBound': 0.9497467283201093}, 0.8: {'lowerBound': 0.9431745186213245, 'upperBound': 0.9518026503284471}, 0.9: {'lowerBound': 0.9419611876000104, 'upperBound': 0.9530159813497613}, 0.95: {'lowerBound': 0.94088267113662, 'upperBound': 0.9540944978131516}, 0.99: {'lowerBound': 0.9387930454888013, 'upperBound': 0.9561841234609704}}, 'runtime': 309.4766981601715},
    # {'optimizing': 'convergence', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.01, 'numFrequentWords': 0, 'accuracy': 0.9388127853881278, 'confidenceIntervals': {0.5: {'lowerBound': 0.9363864122633678, 'upperBound': 0.9412391585128879}, 0.8: {'lowerBound': 0.9341773262841087, 'upperBound': 0.943448244492147}, 0.9: {'lowerBound': 0.9328736034111033, 'upperBound': 0.9447519673651524}, 0.95: {'lowerBound': 0.9317147386350985, 'upperBound': 0.9459108321411572}, 0.99: {'lowerBound': 0.9294694381315892, 'upperBound': 0.9481561326446665}}, 'runtime': 239.19428300857544}
# ]
# evaluations = [
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 1.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9751141552511415, 'confidenceIntervals': {0.5: {'lowerBound': 0.9735371190177567, 'upperBound': 0.9766911914845263}, 0.8: {'lowerBound': 0.9721013099097496, 'upperBound': 0.9781270005925334}, 0.9: {'lowerBound': 0.9712539471574833, 'upperBound': 0.9789743633447998}, 0.95: {'lowerBound': 0.9705007358221353, 'upperBound': 0.9797275746801478}, 0.99: {'lowerBound': 0.9690413888598985, 'upperBound': 0.9811869216423845}}, 'runtime': 3263.5032498836517},
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 3.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9762557077625571, 'confidenceIntervals': {0.5: {'lowerBound': 0.9747143652578463, 'upperBound': 0.9777970502672679}, 0.8: {'lowerBound': 0.9733110534251991, 'upperBound': 0.9792003620999151}, 0.9: {'lowerBound': 0.9724828693928171, 'upperBound': 0.9800285461322971}, 0.95: {'lowerBound': 0.9717467058084777, 'upperBound': 0.9807647097166365}, 0.99: {'lowerBound': 0.9703203888638199, 'upperBound': 0.9821910266612943}}, 'runtime': 2824.1224870681763},
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 5.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9769406392694064, 'confidenceIntervals': {0.5: {'lowerBound': 0.9754211575935792, 'upperBound': 0.9784601209452337}, 0.8: {'lowerBound': 0.9740377489036469, 'upperBound': 0.979843529635166}, 0.9: {'lowerBound': 0.973221310988277, 'upperBound': 0.9806599675505359}, 0.95: {'lowerBound': 0.9724955883968371, 'upperBound': 0.9813856901419757}, 0.99: {'lowerBound': 0.9710895008759223, 'upperBound': 0.9827917776628906}}, 'runtime': 2598.1711769104004},
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 10.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 2176.9077258110046},
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 12.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9757990867579909, 'confidenceIntervals': {0.5: {'lowerBound': 0.9742433581838538, 'upperBound': 0.977354815332128}, 0.8: {'lowerBound': 0.9728269485865051, 'upperBound': 0.9787712249294767}, 0.9: {'lowerBound': 0.9719910347257746, 'upperBound': 0.9796071387902071}, 0.95: {'lowerBound': 0.9712480001829032, 'upperBound': 0.9803501733330785}, 0.99: {'lowerBound': 0.9698083707560898, 'upperBound': 0.9817898027598919}}, 'runtime': 2043.715533733368},
    # {'optimizing': 'stepSize', 'numMutualInformationWords': 1000, 'stepSize': 15.0, 'convergence': 0.0001, 'numFrequentWords': 0, 'accuracy': 0.9764840182648402, 'confidenceIntervals': {0.5: {'lowerBound': 0.9749499246072737, 'upperBound': 0.9780181119224067}, 0.8: {'lowerBound': 0.9735532124712802, 'upperBound': 0.9794148240584002}, 0.9: {'lowerBound': 0.9727289233418416, 'upperBound': 0.9802391131878389}, 0.95: {'lowerBound': 0.9719962218934516, 'upperBound': 0.9809718146362288}, 0.99: {'lowerBound': 0.970576612837196, 'upperBound': 0.9823914236924844}}, 'runtime': 1913.1294980049133}
# ]