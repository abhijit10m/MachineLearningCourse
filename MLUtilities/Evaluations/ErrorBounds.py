import math

_zValues = { .5:.67, .68:1.0, .8:1.28, .9:1.64, .95:1.96, .98:2.33, .99:2.58 }


def GetStandardDeviation(mean, sampleSize):
    return math.sqrt((mean) * (1 - mean) / sampleSize)

def GetAccuracyBounds(mean, sampleSize, confidence):
    if mean < 0.0 or mean > 1.0:
        raise UserWarning("mean must be between 0 and 1")

    if sampleSize <= 0:
        raise UserWarning("sampleSize should be positive")

    s_dev = GetStandardDeviation(mean, sampleSize)
    lower = mean - _zValues[confidence] * s_dev
    upper = mean + _zValues[confidence] * s_dev

    return (lower, upper)
    