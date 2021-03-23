import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import numpy as np

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

doModelEvaluation = False
if doModelEvaluation:
    ######
    ### Build a model and evaluate on validation data
    stepSize = 1.0
    convergence = 0.001

    yTrain      = np.asarray(yTrain)
    yValidate   = np.asarray(yValidate)
    yTest       = np.asarray(yTest)

    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

    xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
    xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
    xTest       = np.asarray(featurizer.Featurize(xTestRaw))

    frequentModel = LogisticRegression.LogisticRegression()
    frequentModel.fit(xTrain, yTrain, convergence=convergence, stepSize=stepSize, verbose=True)

    ######
    ### Use equation 5.1 from Mitchell to bound the validation set error and the true error
    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, frequentModel.predict(xValidate))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    ### Compare to most common class model here...


    mostCommonModel = MostCommonClassModel.MostCommonClassModel()
    mostCommonModel.fit(xTrainRaw, yTrain)

    print("Most Common Model :")
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, mostCommonModel.predict(xValidateRaw))
    print("Validation set accuracy: %.4f." % (validationSetAccuracy))
    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(validationSetAccuracy, len(xValidate), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))


# Set this to true when you've completed the previous steps and are ready to move on...
doCrossValidation = True
if doCrossValidation:

    stepSize = 1.0
    convergence = 0.001

    import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation    
    numberOfFolds = 5

    numberOfCorrect = 0
    numberOfInCorrect = 0

    for foldId in range(numberOfFolds):
        (xTrainRawK, yTrainK, xEvaluateRawK, yEvaluateK) = CrossValidation.CrossValidation(xTrainRaw + xValidateRaw, yTrain + yValidate, numberOfFolds, foldId)
                
        featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
        featurizer.CreateVocabulary(xTrainRawK, yTrainK, numMutualInformationWords = 25)

        xTrainK      = np.asarray(featurizer.Featurize(xTrainRawK))
        xEvaluateK   = np.asarray(featurizer.Featurize(xEvaluateRawK))
        yTrainK      = np.asarray(yTrainK)

        frequentModel = LogisticRegression.LogisticRegression()
        frequentModel.fit(xTrainK, yTrainK, convergence=convergence, stepSize=stepSize, verbose=True)

        yPredictK = frequentModel.predict(xEvaluateK)

        for i in range(len(yPredictK)):
            if yEvaluateK[i] == yPredictK[i]:
                numberOfCorrect += 1
            else: 
                numberOfInCorrect += 1

    
    if(numberOfCorrect + numberOfInCorrect != len(xTrainRaw + xValidateRaw)):
        raise UserWarning("Incorrect accuracy detected")
    fold_accuracy = numberOfCorrect / len(xTrainRaw + xValidateRaw)

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

    print("Logistic regression with 25 features by mutual information:")

    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(fold_accuracy, len(xTrainRaw + xValidateRaw), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    numberOfCorrect = 0
    numberOfInCorrect = 0

    for foldId in range(numberOfFolds):
        (xTrainRawK, yTrainK, xEvaluateRawK, yEvaluateK) = CrossValidation.CrossValidation(xTrainRaw + xValidateRaw, yTrain + yValidate, numberOfFolds, foldId)

        xTrainK      = np.asarray(xTrainRawK)
        xEvaluateK   = np.asarray(xEvaluateRawK)
        yTrainK      = np.asarray(yTrainK)

        mostCommonModel = MostCommonClassModel.MostCommonClassModel()
        mostCommonModel.fit(xTrainK, yTrainK)

        yPredictK = mostCommonModel.predict(xEvaluateK)

        for i in range(len(yPredictK)):
            if yEvaluateK[i] == yPredictK[i]:
                numberOfCorrect += 1
            else: 
                numberOfInCorrect += 1
    

    if(numberOfCorrect + numberOfInCorrect != len(xTrainRaw + xValidateRaw)):
        raise UserWarning("Incorrect accuracy detected")

    fold_accuracy = numberOfCorrect / len(xTrainRaw + xValidateRaw)

    import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds


    print("Most Common Model :")

    for confidence in [.5, .8, .9, .95, .99]:
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(fold_accuracy, len(xTrainRaw + xValidateRaw), confidence)    
        print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))

    
    # Good luck!