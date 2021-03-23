kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module3/Assignment2/"

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D
import multiprocessing.dummy as mp
import numpy as np
import logging
import math
from joblib import Parallel, delayed
import MachineLearningCourse.MLUtilities.Learners.NeuralNetworkFullyConnected as NeuralNetworkFullyConnected
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

UNIT_TEST = False
if UNIT_TEST:
    generator = SampleUniform2D.SampleUniform2D(seed=100)
    conceptSquare = ConceptSquare2D.ConceptSquare2D(width=.1)
    conceptLinear = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.5, -0.5])
    conceptCircle = ConceptCircle2D.ConceptCircle2D(radius=.2)

    concept = ConceptCompound2D.ConceptCompound2D(concepts = [ conceptLinear, conceptCircle, conceptSquare ])

    xTestConcept = generator.generate(1000)
    yTestConcept = concept.predict(xTestConcept)

    xTrainConcept = generator.generate(1000)
    yTrainConcept = concept.predict(xTrainConcept)

    print(np.shape(xTrainConcept))
    print(np.shape(yTrainConcept))

    hiddenStructure = [10, 10]

    'x'.join(map(str, hiddenStructure))

    visualizeTrain = Visualize2D.Visualize2D(kOutputDirectory, "NeuralNetworkConcept_Trainset_{}".format('x'.join(map(str, hiddenStructure))))
    visualizeTest = Visualize2D.Visualize2D(kOutputDirectory, "NeuralNetworkConcept_Testset_{}".format('x'.join(map(str, hiddenStructure))))

    model = NeuralNetworkFullyConnected.NeuralNetworkFullyConnected(len(xTrainConcept[0]), hiddenLayersNodeCounts=hiddenStructure)

    model.fit(xTrainConcept, yTrainConcept, maxEpochs=1000, stepSize=5.0, convergence=0.0001)
    # you can use this to visualize what your model is learning.

    # Training Set Accuracy
    visualizeTrain.Plot2DDataAndBinaryConcept(xTrainConcept,yTrainConcept,model)
    visualizeTrain.Save()

    yPredict = model.predict(xTrainConcept)
    EvaluateBinaryClassification.ExecuteAll(yTrainConcept, yPredict)

    # Testing Set Accuracy
    visualizeTest.Plot2DDataAndBinaryConcept(xTestConcept,yTestConcept,model)
    visualizeTest.Save()

    yPredict = model.predict(xTestConcept)
    EvaluateBinaryClassification.ExecuteAll(yTestConcept, yPredict)


(xRaw, yRaw) = BlinkDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize

featurizer = BlinkFeaturize.BlinkFeaturize()

sampleStride = 2
featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False, includeIntensities=True, intensitiesSampleStride=sampleStride)

xTrain    = featurizer.Featurize(xTrainRaw, normalize=True)
xValidate = featurizer.Featurize(xValidateRaw, normalize=True)
xTest     = featurizer.Featurize(xTestRaw, normalize=True)

# for i in range(10):
#     print("%d: %d - " % (i, yTrain[i]), xTrain[i])


import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

import time

from joblib import Parallel, delayed

import PIL
from PIL import Image


    ## Evaluate things...

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
            yPredict =  model.predict(xValidate, classificationThreshold=threshold)
            FPR = EvaluateBinaryClassification.FalsePositiveRate(yValidate, yPredict)
            FNR = EvaluateBinaryClassification.FalseNegativeRate(yValidate, yPredict)
            logging.debug("yPredict: %s", yPredict)
            logging.debug("yValidate: %s", yValidate)
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


def plotValidationVsTrainingSetAccuracy(evaluations, sweepName):
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

    for evaluation in evaluations:
        
        # Training Set Accuracy
        accuracy = evaluation['trainingSetEvaluation']['accuracy']
        (lowerBound, upperBound) = (evaluation['trainingSetEvaluation']['lowerBound'], evaluation['trainingSetEvaluation']['upperBound'])
        yValues_train.append(accuracy)
        errorBars_train.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        # Validation Set Accuracy
        accuracy = evaluation['validationSetEvaluation']['accuracy']
        (lowerBound, upperBound) = (evaluation['validationSetEvaluation']['lowerBound'], evaluation['validationSetEvaluation']['upperBound'])
        yValues_validate.append(accuracy)
        errorBars_validate.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        if sweepName == 'hiddenStructure':
            xValues.append( 'x'.join(map(str, evaluation['hyperparameterValue'])))
        else:
            xValues.append(evaluation['hyperparameterValue'])
        print("yValues_train %", yValues_train)
        print("yValues_validate %", yValues_validate)
        print("errorBars_train %", errorBars_train)
        print("errorBars_validate %", errorBars_validate)
        print("seriesName %", seriesName)
        print("xValues %", xValues)
        Charting.PlotSeriesWithErrorBars([yValues_train, yValues_validate], [errorBars_train, errorBars_validate], seriesName, xValues, chartTitle="Validation Set vs Training Set accuracy", xAxisTitle=sweepName, yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="ValidationSetAccuracy_{}".format(sweepName))


bestParameters = dict()
sweeps = dict()

improvement_xValues = [0, 1, 2, 3, 4, 5, 6, 7, 8]
min_accuracy_improvement = 1

def evaluateNeuralNetworkFullyConnected(runSpecification):
    model = NeuralNetworkFullyConnected.NeuralNetworkFullyConnected(len(xTrain[0]), hiddenLayersNodeCounts=runSpecification['hiddenStructure'])
    model.fit(xTrain, yTrain, maxEpochs=runSpecification['maxEpochs'], stepSize=runSpecification['stepSize'], convergence=runSpecification['convergence'])

    # Training Set Accuracy
    yPredict = model.predict(xTrain)
    EvaluateBinaryClassification.ExecuteAll(yTrain, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yTrain, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTrain), .5)

    trainingSetEvaluation = {
        'set': 'training',
        'accuracy' : accuracy,
        'lowerBound' : lowerBound,
        'upperBound' : upperBound
    }

    # Validation Set Accuracy
    yPredict = model.predict(xValidate)
    EvaluateBinaryClassification.ExecuteAll(yValidate, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yValidate, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidate), .5)

    validationSetEvaluation = {
        'set' : 'validation',
        'accuracy' : accuracy,
        'lowerBound' : lowerBound,
        'upperBound' : upperBound
    }

    # Test Set Accuracy
    yPredict = model.predict(xTest)
    EvaluateBinaryClassification.ExecuteAll(yTest, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yValidate, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidate), .5)

    testingSetEvaluation = {
        'set' : 'testing',
        'accuracy' : accuracy,
        'lowerBound' : lowerBound,
        'upperBound' : upperBound
    }

    hValue = runSpecification[runSpecification['optimizing']]

    if runSpecification['optimizing'] == 'hiddenStructure':
        hValue = math.prod(runSpecification['hiddenStructure'])

    evaluation = {
        'sweep' : runSpecification['optimizing'],
        'hValue' : hValue,
        'hyperparameterValue' : runSpecification[runSpecification['optimizing']],
        'trainingSetEvaluation' : trainingSetEvaluation,
        'validationSetEvaluation' : validationSetEvaluation,
        'testingSetEvaluation' : testingSetEvaluation
    }

    print("Evaluation: {}".format(evaluation))

    if runSpecification['optimizing'] == 'hiddenStructure':
        return (math.prod(runSpecification['hiddenStructure']), evaluation)


    return (runSpecification[runSpecification['optimizing']], evaluation)


def updatebestParameters(evaluations):
    bestEvaluation = evaluations[0]
    for evaluation in evaluations:
        if evaluation['validationSetEvaluation']['accuracy'] > bestEvaluation['validationSetEvaluation']['accuracy']:
            bestEvaluation = evaluation

    if bestEvaluation['validationSetEvaluation']['accuracy'] > bestParameters['validationSetAccuracy']:
        bestParameters[bestEvaluation['sweep']] = bestEvaluation['hyperparameterValue']
        bestParameters['validationSetAccuracy'] = bestEvaluation['validationSetEvaluation']['accuracy']

    print("bestParameters {}".format(bestParameters))


bestParameters['stepSize'] = 1.0
bestParameters['convergence'] = 0.001
bestParameters['maxEpochs'] = 1000
bestParameters['hiddenStructure'] = [10]
bestParameters['validationSetAccuracy'] = 0.5

SINGLE_LAYER_TUNING = True
if SINGLE_LAYER_TUNING:
    ## Single Layer Tuning

    sweeps['stepSize'] = [4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
    sweeps['convergence'] = [ 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    sweeps['hiddenStructure'] = [[5], [6], [7], [8], [9], [10], [11], [12]]

    evaluationRunSpecifications = []
    for hiddenStructure in sweeps['hiddenStructure']:

        runSpecification = {}
        runSpecification['optimizing'] = 'hiddenStructure'
        runSpecification['stepSize'] = bestParameters['stepSize']
        runSpecification['convergence'] = bestParameters['convergence']
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = hiddenStructure
        evaluationRunSpecifications.append(runSpecification)

    modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    print("Evaluations: ")
    evaluations = [x[1] for x in modelOutputs]

    # evaluations = [{'sweep': 'hiddenStructure', 'hValue': 5, 'hyperparameterValue': [5], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.874859392575928, 'lowerBound': 0.8711417966348847, 'upperBound': 0.8785769885169714}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8561797752808988, 'lowerBound': 0.8450345908124477, 'upperBound': 0.86732495974935}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48764044943820223, 'lowerBound': 0.47176477893382374, 'upperBound': 0.5035161199425807}}, {'sweep': 'hiddenStructure', 'hValue': 6, 'hyperparameterValue': [6], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8655793025871766, 'lowerBound': 0.8617468178822105, 'upperBound': 0.8694117872921427}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8629213483146068, 'lowerBound': 0.8519977602082262, 'upperBound': 0.8738449364209874}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'hiddenStructure', 'hValue': 7, 'hyperparameterValue': [7], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8262092238470191, 'lowerBound': 0.8219517468758792, 'upperBound': 0.830466700818159}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.7955056179775281, 'lowerBound': 0.7826953824884634, 'upperBound': 0.8083158534665927}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4786516853932584, 'lowerBound': 0.4627856440948998, 'upperBound': 0.49451772669161703}}, {'sweep': 'hiddenStructure', 'hValue': 8, 'hyperparameterValue': [8], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.843644544431946, 'lowerBound': 0.8395638864382213, 'upperBound': 0.8477252024256707}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8247191011235955, 'lowerBound': 0.8126433231428825, 'upperBound': 0.8367948791043085}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'hiddenStructure', 'hValue': 9, 'hyperparameterValue': [9], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8208661417322834, 'lowerBound': 0.816557712757537, 'upperBound': 0.8251745707070299}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8157303370786517, 'lowerBound': 0.8034164540190911, 'upperBound': 0.8280442201382123}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.47191011235955055, 'lowerBound': 0.45605466994217814, 'upperBound': 0.48776555477692296}}, {'sweep': 'hiddenStructure', 'hValue': 10, 'hyperparameterValue': [10], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8692350956130483, 'lowerBound': 0.86544711146212, 'upperBound': 0.8730230797639768}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.851685393258427, 'lowerBound': 0.8403971500204069, 'upperBound': 0.862973636496447}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4651685393258427, 'lowerBound': 0.449326596649034, 'upperBound': 0.48101048200265134}}, {'sweep': 'hiddenStructure', 'hValue': 11, 'hyperparameterValue': [11], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8419572553430821, 'lowerBound': 0.8378587431633056, 'upperBound': 0.8460557675228586}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8382022471910112, 'lowerBound': 0.8265057599786251, 'upperBound': 0.8498987344033974}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49887640449438203, 'lowerBound': 0.4829959215765115, 'upperBound': 0.5147568874122526}}, {'sweep': 'hiddenStructure', 'hValue': 12, 'hyperparameterValue': [12], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8605174353205849, 'lowerBound': 0.8566248895696461, 'upperBound': 0.8644099810715237}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8471910112359551, 'lowerBound': 0.8357632826353995, 'upperBound': 0.8586187398365107}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4943820224719101, 'lowerBound': 0.4785025019200628, 'upperBound': 0.5102615430237574}}]

    evaluations = sorted(evaluations, key = lambda x: math.prod(x['hyperparameterValue']))
    print(evaluations)
    plotValidationVsTrainingSetAccuracy(evaluations, "hiddenStructure")

    updatebestParameters(evaluations)

    evaluationRunSpecifications = []
    for convergence in sweeps['convergence']:

        runSpecification = {}
        runSpecification['optimizing'] = 'convergence'
        runSpecification['stepSize'] = bestParameters['stepSize']
        runSpecification['convergence'] = convergence
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = bestParameters['hiddenStructure']
        evaluationRunSpecifications.append(runSpecification)

    modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    evaluations = [x[1] for x in modelOutputs]
    # evaluations = [{'sweep': 'convergence', 'hValue': 0.0001, 'hyperparameterValue': 0.0001, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8655793025871766, 'lowerBound': 0.8617468178822105, 'upperBound': 0.8694117872921427}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8629213483146068, 'lowerBound': 0.8519977602082262, 'upperBound': 0.8738449364209874}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'convergence', 'hValue': 0.0005, 'hyperparameterValue': 0.0005, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8655793025871766, 'lowerBound': 0.8617468178822105, 'upperBound': 0.8694117872921427}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8629213483146068, 'lowerBound': 0.8519977602082262, 'upperBound': 0.8738449364209874}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'convergence', 'hValue': 0.001, 'hyperparameterValue': 0.001, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8655793025871766, 'lowerBound': 0.8617468178822105, 'upperBound': 0.8694117872921427}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8629213483146068, 'lowerBound': 0.8519977602082262, 'upperBound': 0.8738449364209874}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'convergence', 'hValue': 0.005, 'hyperparameterValue': 0.005, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8613610798650169, 'lowerBound': 0.8574784219145938, 'upperBound': 0.86524373781544}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8719101123595505, 'lowerBound': 0.8612958925576923, 'upperBound': 0.8825243321614087}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'convergence', 'hValue': 0.01, 'hyperparameterValue': 0.01, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8613610798650169, 'lowerBound': 0.8574784219145938, 'upperBound': 0.86524373781544}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8719101123595505, 'lowerBound': 0.8612958925576923, 'upperBound': 0.8825243321614087}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'convergence', 'hValue': 0.05, 'hyperparameterValue': 0.05, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8534870641169854, 'lowerBound': 0.84951395599737, 'upperBound': 0.8574601722366008}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8337078651685393, 'lowerBound': 0.821881872623141, 'upperBound': 0.8455338577139376}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49887640449438203, 'lowerBound': 0.4829959215765115, 'upperBound': 0.5147568874122526}}, {'sweep': 'convergence', 'hValue': 0.1, 'hyperparameterValue': 0.1, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8534870641169854, 'lowerBound': 0.84951395599737, 'upperBound': 0.8574601722366008}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8337078651685393, 'lowerBound': 0.821881872623141, 'upperBound': 0.8455338577139376}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49887640449438203, 'lowerBound': 0.4829959215765115, 'upperBound': 0.5147568874122526}}, {'sweep': 'convergence', 'hValue': 0.5, 'hyperparameterValue': 0.5, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8534870641169854, 'lowerBound': 0.84951395599737, 'upperBound': 0.8574601722366008}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8337078651685393, 'lowerBound': 0.821881872623141, 'upperBound': 0.8455338577139376}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49887640449438203, 'lowerBound': 0.4829959215765115, 'upperBound': 0.5147568874122526}}]
    evaluations = sorted(evaluations, key = lambda x: x['hyperparameterValue'])
    
    print("Evaluations: ")
    print(evaluations)
    plotValidationVsTrainingSetAccuracy(evaluations, "convergence")

    updatebestParameters(evaluations)


    evaluationRunSpecifications = []
    for stepSize in sweeps['stepSize']:

        runSpecification = {}
        runSpecification['optimizing'] = 'stepSize'
        runSpecification['stepSize'] = stepSize
        runSpecification['convergence'] = bestParameters['convergence']
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = bestParameters['hiddenStructure']
        evaluationRunSpecifications.append(runSpecification)


    modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    evaluations = [x[1] for x in modelOutputs]
    # evaluations = [{'sweep': 'stepSize', 'hValue': 0.01, 'hyperparameterValue': 0.01, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8641732283464567, 'lowerBound': 0.8603238817244578, 'upperBound': 0.8680225749684556}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8561797752808988, 'lowerBound': 0.8450345908124477, 'upperBound': 0.86732495974935}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4943820224719101, 'lowerBound': 0.4785025019200628, 'upperBound': 0.5102615430237574}}, {'sweep': 'stepSize', 'hValue': 0.05, 'hyperparameterValue': 0.05, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8554555680539933, 'lowerBound': 0.8515046926067429, 'upperBound': 0.8594064435012436}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8606741573033708, 'lowerBound': 0.8496757444509064, 'upperBound': 0.8716725701558352}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4966292134831461, 'lowerBound': 0.4807490513474698, 'upperBound': 0.5125093756188224}}, {'sweep': 'stepSize', 'hValue': 0.1, 'hyperparameterValue': 0.1, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8546119235095613, 'lowerBound': 0.8506514893672782, 'upperBound': 0.8585723576518444}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8539325842696629, 'lowerBound': 0.842715415275529, 'upperBound': 0.8651497532637968}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4853932584269663, 'lowerBound': 0.4695195132964963, 'upperBound': 0.5012670035574363}}, {'sweep': 'stepSize', 'hValue': 0.5, 'hyperparameterValue': 0.5, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8416760404949382, 'lowerBound': 0.8375745686927774, 'upperBound': 0.845777512297099}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8471910112359551, 'lowerBound': 0.8357632826353995, 'upperBound': 0.8586187398365107}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'stepSize', 'hValue': 1.0, 'hyperparameterValue': 1.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8613610798650169, 'lowerBound': 0.8574784219145938, 'upperBound': 0.86524373781544}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8719101123595505, 'lowerBound': 0.8612958925576923, 'upperBound': 0.8825243321614087}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'stepSize', 'hValue': 2.0, 'hyperparameterValue': 2.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.796400449943757, 'lowerBound': 0.7918761844550701, 'upperBound': 0.8009247154324439}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.7932584269662921, 'lowerBound': 0.780396203443187, 'upperBound': 0.8061206504893972}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48089887640449436, 'lowerBound': 0.46502994573081846, 'upperBound': 0.49676780707817025}}, {'sweep': 'stepSize', 'hValue': 3.0, 'hyperparameterValue': 3.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8051181102362205, 'lowerBound': 0.8006676034707367, 'upperBound': 0.8095686170017042}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.7932584269662921, 'lowerBound': 0.780396203443187, 'upperBound': 0.8061206504893972}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.5101123595505618, 'lowerBound': 0.4942350847462808, 'upperBound': 0.5259896343548427}}, {'sweep': 'stepSize', 'hValue': 4.0, 'hyperparameterValue': 4.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.7516872890888638, 'lowerBound': 0.7468331523570325, 'upperBound': 0.7565414258206952}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.7258426966292135, 'lowerBound': 0.7116744547420832, 'upperBound': 0.7400109385163438}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4764044943820225, 'lowerBound': 0.4605416641179398, 'upperBound': 0.4922673246461052}}]
    evaluations = sorted(evaluations, key = lambda x: x['hyperparameterValue'])

    print("Evaluations: ")
    print(evaluations)

    plotValidationVsTrainingSetAccuracy(evaluations, "stepSize")
    updatebestParameters(evaluations)

    print(bestParameters)
    ## Best single layer parameters : 
    # bestParameters = {'stepSize': 1.0, 'convergence': 0.005, 'maxEpochs': 1000, 'hiddenStructure': [6], 'validationSetAccuracy': 0.8719101123595505}


DOUBLE_LAYER_TUNING = False
if DOUBLE_LAYER_TUNING:
    ## Double Layer Tuning

    sweeps['stepSize'] = [4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
    sweeps['convergence'] = [ 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    sweeps['hiddenStructure'] = [[2,2], [3,2], [4,2], [5,2], [6,2], [7,2], [8,2], [9,2]]

    evaluationRunSpecifications = []
    for hiddenStructure in sweeps['hiddenStructure']:

        runSpecification = {}
        runSpecification['optimizing'] = 'hiddenStructure'
        runSpecification['stepSize'] = bestParameters['stepSize']
        runSpecification['convergence'] = bestParameters['convergence']
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = hiddenStructure
        evaluationRunSpecifications.append(runSpecification)

    # modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    # evaluations = [x[1] for x in modelOutputs]
    print("Evaluations: ")
    evaluations = [{'sweep': 'hiddenStructure', 'hValue': 4, 'hyperparameterValue': [2, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8287401574803149, 'lowerBound': 0.8245073268768472, 'upperBound': 0.8329729880837826}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.802247191011236, 'lowerBound': 0.7895966170412633, 'upperBound': 0.8148977649812087}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49887640449438203, 'lowerBound': 0.4829959215765115, 'upperBound': 0.5147568874122526}}, {'sweep': 'hiddenStructure', 'hValue': 6, 'hyperparameterValue': [3, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8498312710911136, 'lowerBound': 0.8458175237325247, 'upperBound': 0.8538450184497024}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8337078651685393, 'lowerBound': 0.821881872623141, 'upperBound': 0.8455338577139376}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48089887640449436, 'lowerBound': 0.46502994573081846, 'upperBound': 0.49676780707817025}}, {'sweep': 'hiddenStructure', 'hValue': 8, 'hyperparameterValue': [4, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8464566929133859, 'lowerBound': 0.8424061639364515, 'upperBound': 0.8505072218903202}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8337078651685393, 'lowerBound': 0.821881872623141, 'upperBound': 0.8455338577139376}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48764044943820223, 'lowerBound': 0.47176477893382374, 'upperBound': 0.5035161199425807}}, {'sweep': 'hiddenStructure', 'hValue': 10, 'hyperparameterValue': [5, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8712035995500562, 'lowerBound': 0.8674399808305056, 'upperBound': 0.8749672182696069}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8831460674157303, 'lowerBound': 0.8729429535530041, 'upperBound': 0.8933491812784565}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.47191011235955055, 'lowerBound': 0.45605466994217814, 'upperBound': 0.48776555477692296}}, {'sweep': 'hiddenStructure', 'hValue': 12, 'hyperparameterValue': [6, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8728908886389202, 'lowerBound': 0.8691483848564097, 'upperBound': 0.8766333924214307}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8539325842696629, 'lowerBound': 0.842715415275529, 'upperBound': 0.8651497532637968}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'hiddenStructure', 'hValue': 14, 'hyperparameterValue': [7, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8818897637795275, 'lowerBound': 0.878263621450884, 'upperBound': 0.8855159061081711}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8853932584269663, 'lowerBound': 0.8752758802168648, 'upperBound': 0.8955106366370678}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4651685393258427, 'lowerBound': 0.449326596649034, 'upperBound': 0.48101048200265134}}, {'sweep': 'hiddenStructure', 'hValue': 16, 'hyperparameterValue': [8, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8799212598425197, 'lowerBound': 0.8762691074298743, 'upperBound': 0.8835734122551651}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.851685393258427, 'lowerBound': 0.8403971500204069, 'upperBound': 0.862973636496447}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'hiddenStructure', 'hValue': 18, 'hyperparameterValue': [9, 2], 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8419572553430821, 'lowerBound': 0.8378587431633056, 'upperBound': 0.8460557675228586}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8404494382022472, 'lowerBound': 0.828818901350999, 'upperBound': 0.8520799750534953}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4898876404494382, 'lowerBound': 0.47401036564515725, 'upperBound': 0.5057649152537191}}]

    evaluations = sorted(evaluations, key = lambda x: math.prod(x['hyperparameterValue']))
    print(evaluations)
    plotValidationVsTrainingSetAccuracy(evaluations, "hiddenStructure")

    updatebestParameters(evaluations)

    evaluationRunSpecifications = []
    for convergence in sweeps['convergence']:

        runSpecification = {}
        runSpecification['optimizing'] = 'convergence'
        runSpecification['stepSize'] = bestParameters['stepSize']
        runSpecification['convergence'] = convergence
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = bestParameters['hiddenStructure']
        evaluationRunSpecifications.append(runSpecification)

    # modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    # evaluations = [x[1] for x in modelOutputs]
    evaluations = [{'sweep': 'convergence', 'hValue': 0.0001, 'hyperparameterValue': 0.0001, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.9097300337457818, 'lowerBound': 0.90651028611012, 'upperBound': 0.9129497813814436}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8876404494382022, 'lowerBound': 0.8776100475047148, 'upperBound': 0.8976708513716896}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4853932584269663, 'lowerBound': 0.4695195132964963, 'upperBound': 0.5012670035574363}}, {'sweep': 'convergence', 'hValue': 0.0005, 'hyperparameterValue': 0.0005, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.890607424071991, 'lowerBound': 0.8871004625037491, 'upperBound': 0.8941143856402329}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8786516853932584, 'lowerBound': 0.8682806994687403, 'upperBound': 0.8890226713177766}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4696629213483146, 'lowerBound': 0.4538116561934996, 'upperBound': 0.4855141865031296}}, {'sweep': 'convergence', 'hValue': 0.001, 'hyperparameterValue': 0.001, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8818897637795275, 'lowerBound': 0.878263621450884, 'upperBound': 0.8855159061081711}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8853932584269663, 'lowerBound': 0.8752758802168648, 'upperBound': 0.8955106366370678}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4651685393258427, 'lowerBound': 0.449326596649034, 'upperBound': 0.48101048200265134}}, {'sweep': 'convergence', 'hValue': 0.005, 'hyperparameterValue': 0.005, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8703599550056242, 'lowerBound': 0.8665858588257235, 'upperBound': 0.874134051185525}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8359550561797753, 'lowerBound': 0.8241934214923107, 'upperBound': 0.8477166908672399}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}, {'sweep': 'convergence', 'hValue': 0.01, 'hyperparameterValue': 0.01, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8636107986501688, 'lowerBound': 0.8597547460293871, 'upperBound': 0.8674668512709505}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8651685393258427, 'lowerBound': 0.8543207616403878, 'upperBound': 0.8760163170112976}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4764044943820225, 'lowerBound': 0.4605416641179398, 'upperBound': 0.4922673246461052}}, {'sweep': 'convergence', 'hValue': 0.05, 'hyperparameterValue': 0.05, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8636107986501688, 'lowerBound': 0.8597547460293871, 'upperBound': 0.8674668512709505}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8651685393258427, 'lowerBound': 0.8543207616403878, 'upperBound': 0.8760163170112976}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4764044943820225, 'lowerBound': 0.4605416641179398, 'upperBound': 0.4922673246461052}}, {'sweep': 'convergence', 'hValue': 0.1, 'hyperparameterValue': 0.1, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8636107986501688, 'lowerBound': 0.8597547460293871, 'upperBound': 0.8674668512709505}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8651685393258427, 'lowerBound': 0.8543207616403878, 'upperBound': 0.8760163170112976}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4764044943820225, 'lowerBound': 0.4605416641179398, 'upperBound': 0.4922673246461052}}, {'sweep': 'convergence', 'hValue': 0.5, 'hyperparameterValue': 0.5, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8636107986501688, 'lowerBound': 0.8597547460293871, 'upperBound': 0.8674668512709505}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8651685393258427, 'lowerBound': 0.8543207616403878, 'upperBound': 0.8760163170112976}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4764044943820225, 'lowerBound': 0.4605416641179398, 'upperBound': 0.4922673246461052}}]

    evaluations = sorted(evaluations, key = lambda x: x['hyperparameterValue'])
    
    print("Evaluations: ")
    print(evaluations)
    plotValidationVsTrainingSetAccuracy(evaluations, "convergence")

    updatebestParameters(evaluations)


    evaluationRunSpecifications = []
    for stepSize in sweeps['stepSize']:

        runSpecification = {}
        runSpecification['optimizing'] = 'stepSize'
        runSpecification['stepSize'] = stepSize
        runSpecification['convergence'] = bestParameters['convergence']
        runSpecification['maxEpochs'] = bestParameters['maxEpochs']
        runSpecification['hiddenStructure'] = bestParameters['hiddenStructure']
        evaluationRunSpecifications.append(runSpecification)


    modelOutputs = Parallel(n_jobs=8)(delayed(evaluateNeuralNetworkFullyConnected)(runSpecification) for runSpecification in evaluationRunSpecifications)
    # evaluations = [x[1] for x in modelOutputs]
    # evaluations = sorted(evaluations, key = lambda x: x['hyperparameterValue'])
    
    evaluations = [{'sweep': 'stepSize', 'hValue': 0.01, 'hyperparameterValue': 0.01, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.9308211473565804, 'lowerBound': 0.927970038988993, 'upperBound': 0.9336722557241678}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8966292134831461, 'lowerBound': 0.8869597995992821, 'upperBound': 0.9062986273670101}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.46741573033707867, 'lowerBound': 0.4515689650041636, 'upperBound': 0.48326249566999374}}, {'sweep': 'stepSize', 'hValue': 0.05, 'hyperparameterValue': 0.05, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8959505061867267, 'lowerBound': 0.89252001804821, 'upperBound': 0.8993809943252433}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8741573033707866, 'lowerBound': 0.8636230538673951, 'upperBound': 0.884691552874178}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.46741573033707867, 'lowerBound': 0.4515689650041636, 'upperBound': 0.48326249566999374}}, {'sweep': 'stepSize', 'hValue': 0.1, 'hyperparameterValue': 0.1, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8638920134983127, 'lowerBound': 0.860039311122273, 'upperBound': 0.8677447158743523}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8539325842696629, 'lowerBound': 0.842715415275529, 'upperBound': 0.8651497532637968}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48089887640449436, 'lowerBound': 0.46502994573081846, 'upperBound': 0.49676780707817025}}, {'sweep': 'stepSize', 'hValue': 0.5, 'hyperparameterValue': 0.5, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8655793025871766, 'lowerBound': 0.8617468178822105, 'upperBound': 0.8694117872921427}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.849438202247191, 'lowerBound': 0.8380797779590674, 'upperBound': 0.8607966265353146}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48089887640449436, 'lowerBound': 0.46502994573081846, 'upperBound': 0.49676780707817025}}, {'sweep': 'stepSize', 'hValue': 1.0, 'hyperparameterValue': 1.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.9097300337457818, 'lowerBound': 0.90651028611012, 'upperBound': 0.9129497813814436}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8876404494382022, 'lowerBound': 0.8776100475047148, 'upperBound': 0.8976708513716896}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.4853932584269663, 'lowerBound': 0.4695195132964963, 'upperBound': 0.5012670035574363}}, {'sweep': 'stepSize', 'hValue': 2.0, 'hyperparameterValue': 2.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.876265466816648, 'lowerBound': 0.8725658458077148, 'upperBound': 0.8799650878255811}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8606741573033708, 'lowerBound': 0.8496757444509064, 'upperBound': 0.8716725701558352}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'stepSize', 'hValue': 3.0, 'hyperparameterValue': 3.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8405511811023622, 'lowerBound': 0.836437916376261, 'upperBound': 0.8446644458284633}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8157303370786517, 'lowerBound': 0.8034164540190911, 'upperBound': 0.8280442201382123}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.48314606741573035, 'lowerBound': 0.4672745688500236, 'upperBound': 0.4990175659814371}}, {'sweep': 'stepSize', 'hValue': 4.0, 'hyperparameterValue': 4.0, 'trainingSetEvaluation': {'set': 'training', 'accuracy': 0.8174915635545557, 'lowerBound': 0.8131516903763845, 'upperBound': 0.8218314367327268}, 'validationSetEvaluation': {'set': 'validation', 'accuracy': 0.8134831460674158, 'lowerBound': 0.8011114820353518, 'upperBound': 0.8258548100994798}, 'testingSetEvaluation': {'set': 'validation', 'accuracy': 0.49213483146067416, 'lowerBound': 0.47625627333317705, 'upperBound': 0.5080133895881712}}]
    
    print("Evaluations: ")
    print(evaluations)

    plotValidationVsTrainingSetAccuracy(evaluations, "stepSize")
    updatebestParameters(evaluations)

    print(bestParameters)
    bestParameters  = {'stepSize': 0.01, 'convergence': 0.0001, 'maxEpochs': 1000, 'hiddenStructure': [7, 2], 'validationSetAccuracy': 0.8966292134831461}



VISUALIZE_WEIGHTS = True
if VISUALIZE_WEIGHTS:

    def VisualizeWeights(weightArray, outputPath, sampleStride = 2):
        imageDimension = int(24 / sampleStride)
        pixelSize = 2 * sampleStride
        imageSize = imageDimension * pixelSize

        # note the extra weight for the bias is where the +1 comes from
        if len(weightArray) != (imageDimension * imageDimension) + 1:
            raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (imageDimension * imageDimension) + 1))

        if not outputPath.endswith(".jpg"):
            raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

        image = Image.new("RGB", (imageSize, imageSize), "White")

        pixels = image.load()

        for x in range(imageDimension):
            for y in range(imageDimension):
                weight = weightArray[1+(x*imageDimension) + y]
                
                # Add in the bias to help understand the weight's function
                weight += weightArray[0]
                
                if weight >= 0:
                    color = (0, int(255 * abs(weight)), 0)
                else:
                    color = (int(255 * abs(weight)), 0, 0)
                
                for i in range(pixelSize):
                    for j in range(pixelSize):
                        pixels[(x * pixelSize) + i, (y * pixelSize) + j] = color

        image.save(outputPath) 
        
    model = NeuralNetworkFullyConnected.NeuralNetworkFullyConnected(len(xTrain[0]), hiddenLayersNodeCounts=bestParameters['hiddenStructure'])

    # for filterNumber in range(bestParameters['hiddenStructure'][0]):
    #     ## update the first parameter based on your representation
    #     ## layers[1][filterNumber] is an array of weights
    #     VisualizeWeights(model.layers[0][filterNumber], "%sfilters/epoch%d_neuron%d.jpg" % (kOutputDirectory, 0, filterNumber), sampleStride=sampleStride)

    # Epoch is 1 when you have iterated over all training samples 1 time

    yValues = []
    errorBars = []
    seriesName = ["Training set Loss","Validation set Loss"]
    xValues = []

    yValues_train = []
    yValues_validate = []

    min_lowerBound = 100000

    yLoss = model.loss(xTrain, yTrain)

    if yLoss < min_lowerBound:
        min_lowerBound = yLoss

    yValues_train.append(yLoss)

    yLoss = model.loss(xValidate, yValidate)
    yValues_validate.append(yLoss)
    if yLoss < min_lowerBound:
        min_lowerBound = yLoss
    xValues.append(0)

    for i in range(1, bestParameters['maxEpochs']): 
        if not model.converged:
            model.incrementalFit(xTrain, yTrain, maxEpochs = 1, stepSize=bestParameters['stepSize'], convergence=bestParameters['convergence'])
            yLoss = model.loss(xTrain, yTrain)

            if yLoss < min_lowerBound:
                min_lowerBound = yLoss

            yValues_train.append(yLoss)

            yLoss = model.loss(xValidate, yValidate)
            yValues_validate.append(yLoss)
            if yLoss < min_lowerBound:
                min_lowerBound = yLoss
            xValues.append(i)

            Charting.PlotSeries([yValues_train, yValues_validate], seriesName, xValues, chartTitle="Loss vs Epoch", xAxisTitle="Epoch", yAxisTitle="Loss", yBotLimit=min_lowerBound - bestParameters['convergence'], outputDirectory=kOutputDirectory, fileName="TrainingSetVsValidationSetLoss")

            # for filterNumber in range(bestParameters['hiddenStructure'][0]):
            #     ## update the first parameter based on your representation
            #     VisualizeWeights(model.layers[0][filterNumber], "%sfilters/epoch%d_neuron%d.jpg" % (kOutputDirectory, i+1, filterNumber), sampleStride=sampleStride)

