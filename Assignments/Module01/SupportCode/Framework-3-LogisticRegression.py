import numpy as np

kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/"

runUnitTest = False
if runUnitTest:
    # Little synthetic dataset to help with implementation. 2 features, 8 samples.
    xTrain = np.array([[.1, .1], [.2, .2], [.2, .1], [.1, .2], [.95, .95], [.9, .8], [.8, .9], [.7, .6]])
    yTrain = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # create a linear model with the right number of weights initialized
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    model = LogisticRegression.LogisticRegression(featureCount = len(xTrain[0]))

    # To use this visualizer you need to install the PIL imaging library. Instructions are in the lecture notes.
    import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

    while not model.converged:
        # do 10 iterations of training
        model.incrementalFit(xTrain, yTrain, maxSteps=10, stepSize=1.0, convergence=0.005)
        
        # then look at the models weights
        model.visualize()
        
        # then look at how training set loss is converging
        print(" fit for %d iterations, train set loss is %.4f" % (model.totalGradientDescentSteps, model.loss(xTrain, yTrain)))

        # and visualize the model's decision boundary
        visualization = Visualize2D.Visualize2D(kOutputDirectory, "{0:04}.test".format(model.totalGradientDescentSteps))
        visualization.Plot2DDataAndBinaryConcept(xTrain, yTrain, model)
        visualization.Save()
        
# Once your LogisticRegression learner seems to be working, set this flag to True and try it on the spam data
runSMSSpam  = True
if runSMSSpam:
    import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

    ############################
    # Set up the data

    (xRaw, yRaw) = SMSSpamDataset.LoadRawData()
    
    import MachineLearningCourse.MLUtilities.Data.Sample as Sample
    (xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

    import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)
    featurizer.CreateVocabulary(xTrainRaw, yTrain, supplementalVocabularyWords=['call','to','your'])

    xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
    xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
    xTest       = np.asarray(featurizer.Featurize(xTestRaw))

    yTrain      = np.asarray(yTrain)
    yValidate   = np.asarray(yValidate)
    yTest       = np.asarray(yTest)

    #############################
    # Learn the logistic regression model
    
    print("Learning the logistic regression model:")
    import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
    import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification

    logisticRegressionModel = LogisticRegression.LogisticRegression(featureCount = len(xTrain[0]))
    
    # logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.01)
    # print ("\nLogistic regression model:")
    # logisticRegressionModel.visualize()
    # EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    # logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.001)
    # print ("\nLogistic regression model:")
    # logisticRegressionModel.visualize()
    # EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))


    # logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.0001)
    # print ("\nLogistic regression model:")
    # logisticRegressionModel.visualize()
    # EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))


    # logisticRegressionModel.fit(xTrain, yTrain, stepSize=1.0, convergence=0.00001)
    # print ("\nLogistic regression model:")
    # logisticRegressionModel.visualize()
    # EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

    iteration = 0
    trainLosses = []
    validationLosses = []
    lossXLabels = []

    while not logisticRegressionModel.converged:
        iteration += 1
        conv = 0.00001
        # do 10 iterations of training
        logisticRegressionModel.incrementalFit(xTrain, yTrain, maxSteps=100, stepSize=1.0, convergence = conv)
        trainLoss = logisticRegressionModel.loss(xTrain, yTrain)
        validationLoss = logisticRegressionModel.loss(xValidate, yValidate)
        trainLosses.append(trainLoss)
        validationLosses.append(validationLoss)
        # then look at the models weights
        logisticRegressionModel.visualize()
        
        # then look at how training set loss is converging
        print(" fit for %d iterations, train set loss is %.4f" % (logisticRegressionModel.totalGradientDescentSteps, trainLoss))
        print(" fit for %d iterations, validation set loss is %.4f" % (logisticRegressionModel.totalGradientDescentSteps, validationLoss))

        logisticRegressionModel.visualize()
        EvaluateBinaryClassification.ExecuteAll(yValidate, logisticRegressionModel.predict(xValidate, classificationThreshold=0.5))

        lossXLabels.append(logisticRegressionModel.totalGradientDescentSteps)
        Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Logistic Regression with convergence={}".format(conv), xAxisTitle="Gradient Descent Steps", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="3-Logistic Regression Train vs Validate loss")
    

    ################
    # You may find the following module helpful for making charts. You'll have to install matplotlib (see the lecture notes).

    # import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
    
    # trainLosses, validationLosses, and lossXLabels are parallel arrays with the losses you want to plot at the specified x coordinates

    # Charting.PlotSeries([trainLosses, validationLosses], ['Train', 'Validate'], lossXLabels, chartTitle="Logistic Regression", xAxisTitle="Gradient Descent Steps", yAxisTitle="Avg. Loss", outputDirectory=kOutputDirectory, fileName="3-Logistic Regression Train vs Validate loss")
