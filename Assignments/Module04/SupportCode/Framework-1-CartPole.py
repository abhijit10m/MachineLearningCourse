kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module4/Assignment1/"

import gym

env = gym.make('CartPole-v0')

import random
import MachineLearningCourse.MLUtilities.Reinforcement.QLearning as QLearning
import MachineLearningCourse.Assignments.Module04.SupportCode.GymSupport as GymSupport
import logging
import sys
import math
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

from joblib import Parallel, delayed

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')

trainingIterations = 20000

def generatePolicy(discountRate, actionProbabilityBase, randomActionRate, learningRateScale, binsPerDimension):
    continuousToDiscrete = GymSupport.ContinuousToDiscrete(binsPerDimension, [ -4.8000002e+00, -4, -4.1887903e-01, -4 ], [ 4.8000002e+00, 4, 4.1887903e-01, 4 ])

    qlearner = QLearning.QLearning(stateSpaceShape=continuousToDiscrete.StateSpaceShape(), numActions=env.action_space.n, discountRate=discountRate)

    # Learn the policy
    for trialNumber in range(trainingIterations):
        observation = env.reset()
        logger.debug(observation)
        reward = 0
        for i in range(300):
            # env.render()

            currentState = continuousToDiscrete.Convert(observation)
            logger.debug(currentState)
            action = qlearner.GetAction(currentState, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase, learningMode=True)


            oldState = continuousToDiscrete.Convert(observation)
            observation, reward, isDone, info = env.step(action)
            newState = continuousToDiscrete.Convert(observation)

            qlearner.ObserveAction(oldState, action, newState, reward, learningRateScale=learningRateScale)

            if isDone:
                if(trialNumber%1000) == 0:
                    logger.debug("%s, %s, %s", trialNumber, i, reward)
                break
    return qlearner

def evaluatePolicy(qlearner, binsPerDimension, actionProbabilityBase, visualize=False,n = 20):
    continuousToDiscrete = GymSupport.ContinuousToDiscrete(binsPerDimension, [ -4.8000002e+00, -4, -4.1887903e-01, -4 ], [ 4.8000002e+00, 4, 4.1887903e-01, 4 ])
    # Evaluate the policy
    totalRewards = []
    for runNumber in range(n):
        observation = env.reset()
        totalReward = 0
        reward = 0
        for i in range(300):
            if visualize:
                renderDone = env.render()

            currentState = continuousToDiscrete.Convert(observation)
            observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, randomActionRate=0, actionProbabilityBase=actionProbabilityBase, learningMode=False))

            totalReward += reward

            if isDone:
                if visualize:
                    renderDone = env.render()
                    logger.info("%d, %d", i, totalReward)
                totalRewards.append(totalReward)
                break
    return totalRewards


def learnAndEvaluate(runSpecification):
    qlearner = generatePolicy(discountRate=runSpecification['discountRate'], actionProbabilityBase=runSpecification['actionProbabilityBase'], randomActionRate=runSpecification['randomActionRate'], learningRateScale=runSpecification['learningRateScale'], binsPerDimension=runSpecification['binsPerDimension'])
    totalRewards = evaluatePolicy(qlearner, binsPerDimension=runSpecification['binsPerDimension'], actionProbabilityBase=runSpecification['actionProbabilityBase'])
    return (runSpecification, sum(totalRewards)/len(totalRewards), qlearner)

# ## Hyperparameters to tune:
# discountRate = 0.5          
# actionProbabilityBase = math.e  
# randomActionRate = 0.1      
# learningRateScale = 0.01     
# binsPerDimension = 5

bestParameters = dict()

bestParameters['discountRate'] = 0.00001                  # Controls the discount rate for future rewards -- this is gamma from 13.10
bestParameters['actionProbabilityBase'] = math.e          # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
bestParameters['randomActionRate'] = 0.1                  # Percent of time the next action selected by GetAction is totally random
bestParameters['learningRateScale'] = 0.1                 # Should be multiplied by visits_n from 13.11.
bestParameters['binsPerDimension'] = 20
bestParameters['bestAverageReward'] = 0
bestParameters['qlearner'] = None

sweeps = dict()


sweeps['discountRate'] = [0.00001, 0.0001, 0.001, 0.01, 0.1]
sweeps['actionProbabilityBase'] = [math.e - 1.5, math.e - 1.4, math.e - 1.3, math.e - 1.2, math.e - 1.01 , math.e, math.e + 1.01, math.e + 1.2,  math.e + 1.3, math.e + 1.4, math.e + 1.5]
sweeps['randomActionRate'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sweeps['learningRateScale'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
sweeps['binsPerDimension'] = [5, 10, 20, 30, 40, 50]

def plotEvaluations(evaluations):
    min_score = 1000
    yAxisTitle = "Score"
    seriesLabels = evaluations[0][0]['optimizing']
    seriesData = []
    xAxisPoints = []
    xAxisTitle = evaluations[0][0]['optimizing']
    fileName = "scoreVs{}".format(evaluations[0][0]['optimizing'])
    chartTitle = "Plot - Accuracy vs {}".format(evaluations[0][0]['optimizing'])
    for evaluation in evaluations:
        score = evaluation[1]
        seriesData.append(score)
        xAxisPoints.append(evaluation[0][evaluation[0]['optimizing']])
        if score < min_score:
            min_score = score
    print("seriesData: ", seriesData)
    print("xAxisPoints: ", xAxisPoints)

    Charting.PlotSeries([seriesData], [seriesLabels], xAxisPoints, chartTitle=chartTitle, xAxisTitle=xAxisTitle, yAxisTitle=yAxisTitle, yBotLimit=min_score - 10, outputDirectory=kOutputDirectory, fileName=fileName)

def updateBest(evaluations):
    sortedEvaluations = sorted(evaluations, key=lambda evaluation: evaluation[1], reverse=True)
    bestEvaluation = sortedEvaluations[0][0]
    bestAverageReward = sortedEvaluations[0][1]
    bestQlearner = sortedEvaluations[0][2]
    if bestAverageReward > bestParameters['bestAverageReward']:
        bestParameters['bestAverageReward'] = bestAverageReward
        bestParameters[bestEvaluation['optimizing']] = bestEvaluation[bestEvaluation['optimizing']]
        bestParameters['qlearner'] = bestQlearner

    if bestParameters['qlearner'] == None:
        bestParameters['qlearner'] = bestQlearner


OPTIMIZE = True
if OPTIMIZE:
        
    ############### BEGIN OPTIMIZING_



    evaluationRunSpecifications = []

    for binsPerDimension in sweeps['binsPerDimension']:

        runSpecification = {}
        runSpecification['optimizing'] = 'binsPerDimension'
        runSpecification['discountRate'] = bestParameters['discountRate']
        runSpecification['actionProbabilityBase'] = bestParameters['actionProbabilityBase']
        runSpecification['randomActionRate'] = bestParameters['randomActionRate']
        runSpecification['learningRateScale'] = bestParameters['learningRateScale']
        runSpecification['binsPerDimension'] = binsPerDimension
        evaluationRunSpecifications.append(runSpecification)


    evaluations = Parallel(n_jobs=len(evaluationRunSpecifications))(delayed(learnAndEvaluate)(runSpec) for runSpec in evaluationRunSpecifications)
    # evaluations = [({'optimizing': 'binsPerDimension', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.55), ({'optimizing': 'binsPerDimension', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 8}, 149.75), ({'optimizing': 'binsPerDimension', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 10}, 115.65), ({'optimizing': 'binsPerDimension', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 15}, 51.95), ({'optimizing': 'binsPerDimension', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 20}, 144.4)]

    print(evaluations)
    evaluations  = sorted(evaluations, key=lambda evaluation: evaluation[0][evaluation[0]['optimizing']])

    plotEvaluations(evaluations)

    updateBest(evaluations)

    print("bestParameters")
    print(bestParameters)
    ########################


    ########################


    evaluationRunSpecifications = []

    for randomActionRate in sweeps['randomActionRate']:

        runSpecification = {}
        runSpecification['optimizing'] = 'randomActionRate'
        runSpecification['discountRate'] = bestParameters['discountRate']
        runSpecification['actionProbabilityBase'] = bestParameters['actionProbabilityBase']
        runSpecification['randomActionRate'] = randomActionRate
        runSpecification['learningRateScale'] = bestParameters['learningRateScale']
        runSpecification['binsPerDimension'] = bestParameters['binsPerDimension']
        evaluationRunSpecifications.append(runSpecification)


    evaluations = Parallel(n_jobs=len(evaluationRunSpecifications))(delayed(learnAndEvaluate)(runSpec) for runSpec in evaluationRunSpecifications)

    # evaluations = [({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.55), ({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 64.45), ({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.2, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.95), ({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 117.85), ({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.4, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 64.4), ({'optimizing': 'randomActionRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 64.45)]

    print(evaluations)
    evaluations  = sorted(evaluations, key=lambda evaluation: evaluation[0][evaluation[0]['optimizing']])

    plotEvaluations(evaluations)

    updateBest(evaluations)

    print("bestParameters")
    print(bestParameters)
    ########################


    ########################


    evaluationRunSpecifications = []

    for learningRateScale in sweeps['learningRateScale']:

        runSpecification = {}
        runSpecification['optimizing'] = 'learningRateScale'
        runSpecification['discountRate'] = bestParameters['discountRate']
        runSpecification['actionProbabilityBase'] = bestParameters['actionProbabilityBase']
        runSpecification['randomActionRate'] = bestParameters['randomActionRate']
        runSpecification['learningRateScale'] = learningRateScale
        runSpecification['binsPerDimension'] = bestParameters['binsPerDimension']
        evaluationRunSpecifications.append(runSpecification)


    evaluations = Parallel(n_jobs=len(evaluationRunSpecifications))(delayed(learnAndEvaluate)(runSpec) for runSpec in evaluationRunSpecifications)

    # evaluations = [({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 1e-05, 'binsPerDimension': 5}, 10.15), ({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.0001, 'binsPerDimension': 5}, 23.85), ({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.001, 'binsPerDimension': 5}, 13.0), ({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 64.4), ({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 0.1, 'binsPerDimension': 5}, 100.95), ({'optimizing': 'learningRateScale', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 1.0, 'binsPerDimension': 5}, 15.15)]

    print(evaluations)
    evaluations  = sorted(evaluations, key=lambda evaluation: evaluation[0][evaluation[0]['optimizing']])

    plotEvaluations(evaluations)

    updateBest(evaluations)

    print("bestParameters")
    print(bestParameters)

    ########################


    ########################

    evaluationRunSpecifications = []

    for actionProbabilityBase in sweeps['actionProbabilityBase']:

        runSpecification = {}
        runSpecification['optimizing'] = 'actionProbabilityBase'
        runSpecification['discountRate'] = bestParameters['discountRate']
        runSpecification['actionProbabilityBase'] = actionProbabilityBase
        runSpecification['randomActionRate'] = bestParameters['randomActionRate']
        runSpecification['learningRateScale'] = bestParameters['learningRateScale']
        runSpecification['binsPerDimension'] = bestParameters['binsPerDimension']
        evaluationRunSpecifications.append(runSpecification)


    evaluations = Parallel(n_jobs=len(evaluationRunSpecifications))(delayed(learnAndEvaluate)(runSpec) for runSpec in evaluationRunSpecifications)

    # evaluations = [({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 67.5), ({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 1.01, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.55), ({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 1.2, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.95), ({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 1.3, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.55), ({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 1.4, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.55), ({'optimizing': 'actionProbabilityBase', 'discountRate': 0.1, 'actionProbabilityBase': 1.5, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.55)]

    print(evaluations)
    evaluations  = sorted(evaluations, key=lambda evaluation: evaluation[0][evaluation[0]['optimizing']])

    plotEvaluations(evaluations)

    updateBest(evaluations)

    print("bestParameters")
    print(bestParameters)

    ########################


    ########################


    evaluationRunSpecifications = []

    for discountRate in sweeps['discountRate']:

        runSpecification = {}
        runSpecification['optimizing'] = 'discountRate'
        runSpecification['discountRate'] = discountRate
        runSpecification['actionProbabilityBase'] = bestParameters['actionProbabilityBase']
        runSpecification['randomActionRate'] = bestParameters['randomActionRate']
        runSpecification['learningRateScale'] = bestParameters['learningRateScale']
        runSpecification['binsPerDimension'] = bestParameters['binsPerDimension']
        evaluationRunSpecifications.append(runSpecification)


    evaluations = Parallel(n_jobs=len(evaluationRunSpecifications))(delayed(learnAndEvaluate)(runSpec) for runSpec in evaluationRunSpecifications)

    # evaluations = [({'optimizing': 'discountRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.2), ({'optimizing': 'discountRate', 'discountRate': 0.2, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.2), ({'optimizing': 'discountRate', 'discountRate': 0.3, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.2), ({'optimizing': 'discountRate', 'discountRate': 0.4, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.2), ({'optimizing': 'discountRate', 'discountRate': 0.5, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.6), ({'optimizing': 'discountRate', 'discountRate': 0.6, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 66.2), ({'optimizing': 'discountRate', 'discountRate': 0.7, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.6), ({'optimizing': 'discountRate', 'discountRate': 0.8, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.1), ({'optimizing': 'discountRate', 'discountRate': 0.9, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5}, 15.6)]
    print(evaluations)
    evaluations  = sorted(evaluations, key=lambda evaluation: evaluation[0][evaluation[0]['optimizing']])

    plotEvaluations(evaluations)

    updateBest(evaluations)

    print("bestParameters")
    print(bestParameters)
    ########################


    ########################


    print("Tuned Hyperparameters")
    print(bestParameters)

# bestParameters = {'discountRate': 1e-05, 'actionProbabilityBase': 3.918281828459045, 'randomActionRate': 0.1, 'learningRateScale': 0.01, 'binsPerDimension': 5, 'bestAverageReward': 161.75}

totalRewards = evaluatePolicy(bestParameters['qlearner'], binsPerDimension=bestParameters['binsPerDimension'], actionProbabilityBase=bestParameters['actionProbabilityBase'], visualize=True,n=100)

avg_score = sum(totalRewards) / len(totalRewards)
print("average over 100 runs : ", avg_score)



########

[({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 5}, 9.35, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20d30>), ({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 10}, 9.35, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20e50>), ({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20e80>), ({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 30}, 9.75, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd3fc70>), ({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 40}, 9.8, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x1178ed8e0>), ({'optimizing': 'binsPerDimension', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 50}, 9.65, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20eb0>)]
seriesData:  [9.35, 9.35, 10.15, 9.75, 9.8, 9.65]
xAxisPoints:  [5, 10, 20, 30, 40, 50]
bestParameters
{'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 20, 'bestAverageReward': 10.15, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20e80>}
[({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.1, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32eb0>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.2, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32ca0>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.3, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32f40>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.4, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ce6cc40>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32a30>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.6, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32bb0>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.7, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32d00>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.8, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32fd0>), ({'optimizing': 'randomActionRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.9, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32b50>)]
seriesData:  [10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15]
xAxisPoints:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bestParameters
{'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 20, 'bestAverageReward': 10.15, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20e80>}
[({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1e-05, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cea5cd0>), ({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.0001, 'binsPerDimension': 20}, 10.15, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cea5b20>), ({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.001, 'binsPerDimension': 20}, 11.55, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ce5ccd0>), ({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.01, 'binsPerDimension': 20}, 12.8, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ce5cc40>), ({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 0.1, 'binsPerDimension': 20}, 49.4, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ce5cca0>), ({'optimizing': 'learningRateScale', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ced5d00>)]
seriesData:  [10.15, 10.15, 11.55, 12.8, 49.4, 56.0]
xAxisPoints:  [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0]
bestParameters
{'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20, 'bestAverageReward': 56.0, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ced5d00>}
[({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 1.218281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd295e0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 1.3182818284590452, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd292b0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 1.418281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cdc6ca0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 1.5182818284590451, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cf3f340>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 1.708281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd29250>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cda8ee0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 3.7282818284590453, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd290a0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 3.918281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd29d00>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 4.018281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd291c0>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 4.118281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cda8f10>), ({'optimizing': 'actionProbabilityBase', 'discountRate': 1e-05, 'actionProbabilityBase': 4.218281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cdc6af0>)]
seriesData:  [56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0]
xAxisPoints:  [1.218281828459045, 1.3182818284590452, 1.418281828459045, 1.5182818284590451, 1.708281828459045, 2.718281828459045, 3.7282818284590453, 3.918281828459045, 4.018281828459045, 4.118281828459045, 4.218281828459045]
bestParameters
{'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20, 'bestAverageReward': 56.0, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11ced5d00>}
[({'optimizing': 'discountRate', 'discountRate': 1e-05, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20a30>), ({'optimizing': 'discountRate', 'discountRate': 0.0001, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 56.0, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd32970>), ({'optimizing': 'discountRate', 'discountRate': 0.001, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 52.95, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cf8fd60>), ({'optimizing': 'discountRate', 'discountRate': 0.01, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 126.2, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20fd0>), ({'optimizing': 'discountRate', 'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20}, 129.35, <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20610>)]
seriesData:  [56.0, 56.0, 52.95, 126.2, 129.35]
xAxisPoints:  [1e-05, 0.0001, 0.001, 0.01, 0.1]
bestParameters
{'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20, 'bestAverageReward': 129.35, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20610>}
Tuned Hyperparameters
{'discountRate': 0.1, 'actionProbabilityBase': 2.718281828459045, 'randomActionRate': 0.5, 'learningRateScale': 1.0, 'binsPerDimension': 20, 'bestAverageReward': 129.35, 'qlearner': <MachineLearningCourse.MLUtilities.Reinforcement.QLearning.QLearning object at 0x11cd20610>}