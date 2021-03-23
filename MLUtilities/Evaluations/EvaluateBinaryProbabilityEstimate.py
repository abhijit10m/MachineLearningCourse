import math
import numpy

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value < 0 or value > 1:
            valueError = True
    for value in yPredicted:
        if value < 0 or value > 1:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be between 0 and 1.")

def MeanSquaredErrorLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    squaredErrorLoss = [ math.pow(yPredicted[i] - y[i], 2)/2 for i in range(len(y)) ]
    meanSquaredErrorLoss = sum(squaredErrorLoss)/len(y)
    return meanSquaredErrorLoss

def LogLoss(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    log_loss = [ (-1 * y[i] * math.log(float(yPredicted[i])) -  (1 - y[i]) * math.log(1  - float(yPredicted[i])) )  for i in range(len(y))]
    average_log_loss = sum(log_loss)/len(log_loss)
    return average_log_loss