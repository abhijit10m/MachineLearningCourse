import numpy as np

kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize

findTop10Words = False
if findTop10Words:
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)

    print("Top 10 words by frequency: ", featurizer.FindMostFrequentWords(xTrainRaw, 10))
    print("Top 10 words by mutual information: ", featurizer.FindTopWordsByMutualInformation(xTrainRaw, yTrain, 10))

# set to true when your implementation of the 'FindWords' part of the assignment is working
doModeling = True
if doModeling:
    # # Now get into model training
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    # The hyperparameters to use with logistic regression for this assignment
    stepSize = 1.0
    convergence = 0.001

    featuresToTry = [ 50 ]

    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

    trainLosses = []
    validationLosses = []
    lossXLabels = []

    for features in featuresToTry:

        # Remeber to create a new featurizer object/vocabulary for each part of the assignment
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
        # featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = features)
        featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords=features)

        # Remember to reprocess the raw data whenever you change the featurizer    
        xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
        xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
        xTest       = np.asarray(featurizer.Featurize(xTestRaw))

        yTrain      = np.asarray(yTrain)
        yValidate   = np.asarray(yValidate)
        yTest       = np.asarray(yTest)

        #############################
        # Learn the logistic regression model
        
        print("Learning the logistic regression model:")
        logisticRegressionModel = LogisticRegression.LogisticRegression(featureCount = len(xTrain[0]))

        logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.001)
        
        logisticRegressionModel.visualize()

        EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate))
        # trainLosses.append(logisticRegressionModel.loss(xTrain, yTrain))
        # validationLosses.append(logisticRegressionModel.loss(xValidate, yValidate))

        
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    
    # # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates

    # # Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Feature Selection", xAxisTitle="Number of features", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="4-Feature Selection-Top N frequent words")
    Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], featuresToTry, chartTitle="Feature Selection", xAxisTitle="Number of features", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="4-Feature Selection-Top N by mutual information")


        # print("partial linear model")

        # def has_word(word, example):
        #     if word in example:
        #         return True
        #     else:
        #         return False

        # yPredict = []

        # for example in xValidateRaw:
        #     if ((has_word('Call', example) | has_word('call', example)) and (has_word('claim', example)) and has_word('or', example)):
        #         yPredict.append(1)
        #     else:
        #         yPredict.append(0)

        # EvaluateBinaryClassification.ExecuteAll(yValidate, yPredict)

        # for index in range(len(yPredict)):
        #     if yPredict[index] == 1:
        #         print(xValidateRaw[index])


        