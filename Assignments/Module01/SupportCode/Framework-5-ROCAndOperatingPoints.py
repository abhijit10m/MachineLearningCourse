import multiprocessing.dummy as mp
from collections import defaultdict 
import numpy as np
from itertools import repeat
import sys
import logging

kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/"

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

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
         logging.debug("thresholds : %s, FPR: %s, FNR: %s", threshold, FPR, FNR)
      except NotImplementedError:
         raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

      FPR_dict[threshold] = FPR
      FNR_dict[threshold] = FNR
      return (threshold, FPR, FNR)

   results = []
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


# Hyperparameters to use for the run
stepSize = 0.1
convergence = 0.0001


# Set up to hold information for creating ROC curves
seriesFPRs = []
seriesFNRs = []
seriesLabels = []

# #### Learn a model with 25 frequent features
# featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
# featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = 25)

# xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
# xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
# xTest       = np.asarray(featurizer.Featurize(xTestRaw))

# yTrain      = np.asarray(yTrain)
# yValidate   = np.asarray(yValidate)
# yTest       = np.asarray(yTest)

# model = LogisticRegression.LogisticRegression()
# model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

# (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)

# print("thresholds: {}".format(thresholds))
# print("modelFPRs: {}".format(modelFPRs))
# print("modelFNRs: {}".format(modelFNRs))

thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
modelFPRs = [1.0, 1.0, 0.9891304347826086, 0.9586956521739131, 0.9217391304347826, 0.8608695652173913, 0.8108695652173913, 0.7630434782608696, 0.7021739130434783, 0.6217391304347826, 0.5847826086956521, 0.5173913043478261, 0.34130434782608693, 0.30869565217391304, 0.27608695652173915, 0.25, 0.23043478260869565, 0.19782608695652174, 0.15434782608695652, 0.14565217391304347, 0.13043478260869565, 0.11739130434782609, 0.10217391304347827, 0.06739130434782609, 0.06739130434782609, 0.05652173913043478, 0.04782608695652174, 0.043478260869565216, 0.043478260869565216, 0.03695652173913044, 0.03260869565217391, 0.02826086956521739, 0.02826086956521739, 0.015217391304347827, 0.015217391304347827, 0.015217391304347827, 0.015217391304347827, 0.015217391304347827, 0.013043478260869565, 0.008695652173913044, 0.008695652173913044, 0.008695652173913044, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.002173913043478261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
modelFNRs = [0.0, 0.0, 0.0, 0.0, 0.011494252873563218, 0.011494252873563218, 0.022988505747126436, 0.022988505747126436, 0.022988505747126436, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.04597701149425287, 0.05747126436781609, 0.06896551724137931, 0.1724137931034483, 0.20689655172413793, 0.2413793103448276, 0.2413793103448276, 0.25287356321839083, 0.2988505747126437, 0.3218390804597701, 0.3563218390804598, 0.3793103448275862, 0.40229885057471265, 0.4367816091954023, 0.4367816091954023, 0.4367816091954023, 0.45977011494252873, 0.47126436781609193, 0.5057471264367817, 0.5172413793103449, 0.5517241379310345, 0.5862068965517241, 0.5862068965517241, 0.6206896551724138, 0.632183908045977, 0.6436781609195402, 0.6896551724137931, 0.7586206896551724, 0.7701149425287356, 0.7816091954022989, 0.8045977011494253, 0.8160919540229885, 0.8275862068965517, 0.8275862068965517, 0.8390804597701149, 0.8390804597701149, 0.8505747126436781, 0.8735632183908046, 0.8850574712643678, 0.896551724137931, 0.9080459770114943, 0.9195402298850575, 0.9195402298850575, 0.9310344827586207, 0.9310344827586207, 0.9425287356321839, 0.9425287356321839, 0.9425287356321839, 0.9425287356321839, 0.9425287356321839, 0.9425287356321839, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9770114942528736, 0.9770114942528736, 0.9770114942528736, 0.9770114942528736, 0.9770114942528736, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Frequent')

# print("Rate {}".format(list(zip(thresholds, modelFPRs, modelFNRs))))

#### Learn a model with 25 features by mutual information
# featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
# featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

# xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
# xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
# xTest       = np.asarray(featurizer.Featurize(xTestRaw))

# model = LogisticRegression.LogisticRegression()
# model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

# (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)

# print("thresholds: {}".format(thresholds))
# print("modelFPRs: {}".format(modelFPRs))
# print("modelFNRs: {}".format(modelFNRs))

thresholds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
modelFPRs = [1.0, 1.0, 1.0, 1.0, 0.9, 0.8956521739130435, 0.8739130434782608, 0.8130434782608695, 0.8043478260869565, 0.36086956521739133, 0.36086956521739133, 0.33043478260869563, 0.32608695652173914, 0.31956521739130433, 0.24565217391304348, 0.24347826086956523, 0.24347826086956523, 0.09782608695652174, 0.0782608695652174, 0.0782608695652174, 0.07608695652173914, 0.07391304347826087, 0.07391304347826087, 0.06521739130434782, 0.06521739130434782, 0.06521739130434782, 0.02826086956521739, 0.02391304347826087, 0.02391304347826087, 0.02391304347826087, 0.02391304347826087, 0.015217391304347827, 0.013043478260869565, 0.013043478260869565, 0.013043478260869565, 0.013043478260869565, 0.013043478260869565, 0.013043478260869565, 0.013043478260869565, 0.010869565217391304, 0.010869565217391304, 0.010869565217391304, 0.010869565217391304, 0.004347826086956522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
modelFNRs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011494252873563218, 0.011494252873563218, 0.08045977011494253, 0.08045977011494253, 0.09195402298850575, 0.10344827586206896, 0.10344827586206896, 0.11494252873563218, 0.12643678160919541, 0.12643678160919541, 0.14942528735632185, 0.20689655172413793, 0.20689655172413793, 0.21839080459770116, 0.21839080459770116, 0.27586206896551724, 0.28735632183908044, 0.2988505747126437, 0.3448275862068966, 0.367816091954023, 0.39080459770114945, 0.42528735632183906, 0.42528735632183906, 0.42528735632183906, 0.47126436781609193, 0.5057471264367817, 0.5172413793103449, 0.5172413793103449, 0.5287356321839081, 0.5862068965517241, 0.5862068965517241, 0.5862068965517241, 0.5862068965517241, 0.6206896551724138, 0.6436781609195402, 0.6666666666666666, 0.6896551724137931, 0.7471264367816092, 0.7471264367816092, 0.7586206896551724, 0.7586206896551724, 0.7816091954022989, 0.8045977011494253, 0.8045977011494253, 0.8045977011494253, 0.8275862068965517, 0.8505747126436781, 0.8505747126436781, 0.8620689655172413, 0.8620689655172413, 0.8735632183908046, 0.8735632183908046, 0.8735632183908046, 0.8735632183908046, 0.8850574712643678, 0.8850574712643678, 0.8850574712643678, 0.9080459770114943, 0.9195402298850575, 0.9310344827586207, 0.9310344827586207, 0.9425287356321839, 0.9425287356321839, 0.9540229885057471, 0.9540229885057471, 0.9540229885057471, 0.9540229885057471, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9655172413793104, 0.9770114942528736, 0.9770114942528736, 0.9770114942528736, 0.9770114942528736, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 0.9885057471264368, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Mutual Information')


#### Tuning model features by frequent features and Mutual Information

numFrequentWords = 25
numMutualInformationWords = 100
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = numFrequentWords, numMutualInformationWords = numMutualInformationWords)

xTrain      = np.asarray(featurizer.Featurize(xTrainRaw))
xValidate   = np.asarray(featurizer.Featurize(xValidateRaw))
xTest       = np.asarray(featurizer.Featurize(xTestRaw))

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('Tuned model')


print("Rate {}".format(list(zip(thresholds, modelFPRs, modelFNRs))))

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs_Tuning_both_{}_{}".format(numFrequentWords,numMutualInformationWords))