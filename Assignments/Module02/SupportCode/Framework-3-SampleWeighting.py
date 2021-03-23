kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module2/Assignment3/"
import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
import numpy as np
import multiprocessing.dummy as mp
import logging

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")



# some sample tests that call into helper functions in the DecisionTree module. 
#   You may not have implemented the same way, so you might have to adapt these tests.

WeightedEntropyUnitTest = False
if WeightedEntropyUnitTest:
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.Entropy(y, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.Entropy(y, [ 0.1 if label == 1 else 1.0 for label in y ]))


WeightedSplitUnitTest = False
if WeightedSplitUnitTest:
    x = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    y = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.1 if label == 1 else 1.0 for label in y ]))
    
    x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y = [1,1,1,1,1,1,0,0,0,1]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.1 if label == 1 else 1.0 for label in y ]))

    x=[[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y = [1,1,0,1,1,1,1,1,0,1]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.1 if label == 1 else 1.0 for label in y ]))

    x = [[1,3], [2,2], [19,7], [4,1]]
    y = [1,1,0,0]
    
    print("Unweighted: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 1.0 for label in y ]))

    print("All 1s get 0 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.0 if label == 1 else 1.0 for label in y ]))

    print("All 1s get .1 weight: ")
    print(DecisionTreeWeighted.FindBestSplitOnFeature(x, y, 0, [ 0.1 if label == 1 else 1.0 for label in y ]))


WeightTreeUnitTest = False
if WeightTreeUnitTest:
    xTrain = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9], [1.0]]
    yTrain = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]

    print("Unweighted:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, maxDepth = 1)

    model.visualize()

    print("Weighted 1s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, w=[ 10 if y == 1 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()

    print("Weighted 0s:")
    model = DecisionTreeWeighted.DecisionTreeWeighted()
    model.fit(xTrain, yTrain, w=[ 1 if y == 0 else 0.1 for y in yTrain ], maxDepth = 1)

    model.visualize()

Execute = True
if Execute:
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
    import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset

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
                yPredicated = model.predict(xValidate, classificationThreshold=threshold)
                FPR = EvaluateBinaryClassification.FalsePositiveRate(yValidate, yPredicated)
                FNR = EvaluateBinaryClassification.FalseNegativeRate(yValidate, yPredicated)
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



    #######################
    seriesFPRs = []
    seriesFNRs = []
    seriesLabels = []

    (xRaw, yRaw) = AdultDataset.LoadRawData()

    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

    xValidateRaw = np.array(xValidateRaw)
    yValidate = np.array(yValidate)
    xTestRaw = np.array(xTestRaw)
    yTest = np.array(yTest)

    print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
    print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
    print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

    import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize

    import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree

    featurizer = AdultFeaturize.AdultFeaturize()
    featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = True)
    for i in range(featurizer.GetFeatureCount()):
        print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

    xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
    xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))

    # Decision tree weighted equal weights
    modelWeighted = DecisionTreeWeighted.DecisionTreeWeighted()
    wTrain = [ 1.0 for x in xTrain]
    modelWeighted.fit(xTrain.tolist(), yTrain, wTrain, maxDepth=10, verbose=True)

    modelWeighted.visualize()

    (modelWeightedFPRs, modelWeightedFNRs, thresholdsWeighted) = TabulateModelPerformanceForROC(modelWeighted, xValidate, yValidate)
    seriesFPRs.append(modelWeightedFPRs)
    seriesFNRs.append(modelWeightedFNRs)
    seriesLabels.append('Weighted decision tree equal weights')
    print("Rate {}".format(list(zip(thresholdsWeighted, modelWeightedFPRs, modelWeightedFNRs))))

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Adult-DecisionTree-WeightedVsUnweightedDecisionTree_0")

    ## Decision Tree Unweighted
    model = DecisionTree.DecisionTree()
    model.fit(xTrain.tolist(), yTrain, maxDepth=10, verbose=True)

    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
    seriesFPRs.append(modelFPRs)
    seriesFNRs.append(modelFNRs)
    seriesLabels.append('Unweighted decision tree')
    print("Rate {}".format(list(zip(thresholds, modelFPRs, modelFNRs))))

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Adult-DecisionTree-WeightedVsUnweightedDecisionTree_1")

    # ### Decision Tree Weighted
    modelWeighted = DecisionTreeWeighted.DecisionTreeWeighted()
    wTrain = [ 10 if example[0] < 45 else 1 for example in xTrain]
    modelWeighted.fit(xTrain.tolist(), yTrain, wTrain, maxDepth=10, verbose=True)

    modelWeighted.visualize()

    (modelWeightedFPRs, modelWeightedFNRs, thresholdsWeighted) = TabulateModelPerformanceForROC(modelWeighted, xValidate, yValidate)
    seriesFPRs.append(modelWeightedFPRs)
    seriesFNRs.append(modelWeightedFNRs)
    seriesLabels.append('Weighted decision tree')

    Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Adult-DecisionTree-WeightedVsUnweightedDecisionTree_2")
