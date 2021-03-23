from tabulate import tabulate
# This file contains stubs for evaluating binary classifications. You must complete these functions as part of your assignment.
#     Each function takes in: 
#           'y':           the arrary of 0/1 true class labels; 
#           'yPredicted':  the prediction your model made for the cooresponding example.
def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)


# True Positives / (True Positive + False Positive)
def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    try:
        precision = confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[1][0] )
        return precision
    except ZeroDivisionError as e:
        return None

# True Positives / (True Positive + False Negetive)
def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    try:
        recall = confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[0][1] )
        return recall
    except ZeroDivisionError as e:
        return None

# False Negetive / (True Positive + False Negetive)
def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    try:
        false_negetive_rate = confusion_matrix[0][1] / ( confusion_matrix[0][0] + confusion_matrix[0][1] )
        return false_negetive_rate
    except ZeroDivisionError as e:
        return None

# False Positive / ( False Positive + True Negetive)
def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    confusion_matrix = ConfusionMatrix(y, yPredicted)
    try:
        false_positive_rate = confusion_matrix[1][0] / ( confusion_matrix[1][0] + confusion_matrix[1][1] )
        return false_positive_rate
    except ZeroDivisionError as e:
        return None

def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)
    true_positives = 0
    false_positives = 0
    false_negetives = 0
    true_negetives = 0

    for i in range(len(y)):
        if(y[i] == yPredicted[i] == 1):
            true_positives += 1
        elif (y[i] == yPredicted[i] == 0):
            true_negetives += 1
        elif (y[i] == 1 and yPredicted[i] == 0):
            false_negetives +=1
        elif (y[i] == 0 and yPredicted[i] == 1):
            false_positives +=1
    
    return [[true_positives, false_negetives], [false_positives, true_negetives]]


def print_confusion_matrix(c_matrix):
    headers = ["Intelligence says there", "Intelligence says not there"]
    table = [
                ["Someone actually is there",
                    "True positive : {}".format(c_matrix[0][0]),
                    "False negetive : {}".format(c_matrix[0][1])],

                ["Someone actually is not there", 
                    "False positive : {}".format(c_matrix[1][0]),
                    "True negetive : {}".format(c_matrix[1][1])]]

    print(tabulate(table, headers, tablefmt="github"))

def ExecuteAll(y, yPredicted):
    print_confusion_matrix(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    
