kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module2/Assignment4/"

import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Learners.AdaBoost as AdaBoost

import numpy as np
import multiprocessing.dummy as mp
import logging


## remember this helper function
# Charting.PlotSeriesWithErrorBars([yValues], [errorBars], [series names], xValues, chartTitle=", xAxisTitle="", yAxisTitle="", yBotLimit=0.5, outputDirectory=kOutputDirectory, fileName="")

## generate some synthetic data do help debug your learning code

generator = SampleUniform2D.SampleUniform2D(seed=100)
conceptSquare = ConceptSquare2D.ConceptSquare2D(width=.1)
conceptLinear = ConceptLinear2D.ConceptLinear2D(bias=0.05, weights=[0.5, -0.5])
conceptCircle = ConceptCircle2D.ConceptCircle2D(radius=.2)

concept = ConceptCompound2D.ConceptCompound2D(concepts = [ conceptLinear, conceptCircle, conceptSquare ])

xTest = generator.generate(1000)
yTest = concept.predict(xTest)

xTrain = generator.generate(1000)
yTrain = concept.predict(xTrain)


import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

## this code outputs the true concept.
visualize = Visualize2D.Visualize2D(kOutputDirectory, "Generated Concept")
visualize.Plot2DDataAndBinaryConcept(xTest,yTest,concept)
visualize.Save()


# you can use this to visualize what your model is learning.
# visualize = Visualize2D.Visualize2D(kOutputDirectory, "Your Boosted Tree...")
# visualize.PlotBinaryConcept(model)

# Or you can use it to visualize individual models that you learened, e.g.:
# visualize.PlotBinaryConcept(model->modelLearnedInRound[2])
    
## you might like to see the training or test data too, so you might prefer this to simply calling 'PlotBinaryConcept'
# visualize.Plot2DDataAndBinaryConcept(xTrain,yTrain,model)

# And remember to save
# visualize.Save()

rounds = [1, 5, 10, 15, 20, 25, 30, 35, 40]
# rounds = [1, 5]

models = []

yValues = []
errorBars = []
seriesName = "Synthetic concept accuracy"
xValues = rounds

adaBoost = AdaBoost.AdaBoost(rounds[-1], DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=1)
adaBoostModel = adaBoost.adaBoost(xTrain, yTrain)



visualize = Visualize2D.Visualize2D(kOutputDirectory, "Final Boosted Tree Round 30")
visualize.PlotBinaryConcept(adaBoostModel.getModelWithRounds(30))
visualize.Save()

min_lowerBound = 1.0

models = []

for round in rounds:
    models.append(adaBoostModel.getModelWithRounds(round))

for model in models:
    yPredict = model.predict(xTest)
    EvaluateBinaryClassification.ExecuteAll(yTest, yPredict)
    accuracy = EvaluateBinaryClassification.Accuracy(yTest, yPredict)
    (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTest), .5)
    yValues.append(accuracy)
    errorBars.append(accuracy - lowerBound)
    if lowerBound < min_lowerBound:
        min_lowerBound = lowerBound

Charting.PlotSeriesWithErrorBars([yValues], [errorBars], [seriesName], xValues, chartTitle="Accuracy vs Rounds", xAxisTitle="Number of rounds", yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="TestSetAccuracyVsNumberOfRounds")


