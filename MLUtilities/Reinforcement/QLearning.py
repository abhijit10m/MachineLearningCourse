import math
import logging
import sys
import numpy as np
import random

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger('__main__')

random.seed(0)

class QLearning(object):
    """Simple table based q learning"""
    def __init__(self, stateSpaceShape, numActions, discountRate=.5):
        self.numActions = numActions
        self.stateSpaceShape = stateSpaceShape
        self.discountRate = discountRate
        shape = [numActions] + stateSpaceShape
        self.QPred = np.zeros_like(np.ndarray(shape=shape))
        self.visits = np.zeros_like(np.ndarray(shape=shape))
        
        logger.debug(np.shape(self.QPred))

    def GetAction(self, state, randomActionRate, actionProbabilityBase, learningMode=True):
        logger.debug("state %s", state)
        logger.debug(randomActionRate)
        logger.debug(actionProbabilityBase)


        #  Support randomActionRate
        SUPPORT_RANDOM_ACTION_RATE = True
        if SUPPORT_RANDOM_ACTION_RATE and learningMode:
            doRandomize = [True, False]
            randomizeDistribution = [randomActionRate, 1 - randomActionRate]
            shouldRandomize = random.choices(doRandomize, randomizeDistribution)
            if shouldRandomize:
                return random.choice(range(self.numActions))


        if learningMode:
            probability = [ actionProbabilityBase ** self.QPred[i][state[0]][state[1]][state[2]][state[3]] for i in range(self.numActions)]
            logger.debug("probability: %s", probability)

            sumOfProbability = sum(probability)

            conditionalProbability = [ (p) / (sumOfProbability ) for p in probability ]

            return conditionalProbability.index(max(conditionalProbability))

        else:
            probability = [ actionProbabilityBase ** self.QPred[i][state[0]][state[1]][state[2]][state[3]] for i in range(self.numActions)]
            return probability.index(max(probability))

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=0):
        self.visits[action][oldState[0]][oldState[1]][oldState[2]][oldState[3]] += 1
        alphaN = 1 / (1 + learningRateScale * self.visits[action][oldState[0]][oldState[1]][oldState[2]][oldState[3]])

        QPredPrev = self.QPred[action][oldState[0]][oldState[1]][oldState[2]][oldState[3]]
        update = (1 - alphaN) * QPredPrev 
        update += alphaN * (reward + self.discountRate * max( [self.QPred[i][newState[0]][newState[1]][newState[2]][newState[3]] for i in range(self.numActions)] ))

        self.QPred[action][oldState[0]][oldState[1]][oldState[2]][oldState[3]] = update
