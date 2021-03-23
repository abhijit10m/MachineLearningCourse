import os
import random


def LoadRawData(kDataPath, percentForTesting=.2, includeLeftEye = True, includeRightEye = True, shuffle=True):
   xRaw = []
   yRaw = []
    
   if includeLeftEye:
      closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
      for fileName in os.listdir(closedEyeDir):
         if fileName.endswith(".jpg"):
            xRaw.append(os.path.join(closedEyeDir, fileName))
            yRaw.append(1)

      openEyeDir = os.path.join(kDataPath, "openLeftEyes")
      for fileName in os.listdir(openEyeDir):
         if fileName.endswith(".jpg"):
            xRaw.append(os.path.join(openEyeDir, fileName))
            yRaw.append(1)

   if includeRightEye:
      closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
      for fileName in os.listdir(closedEyeDir):
         if fileName.endswith(".jpg"):
            xRaw.append(os.path.join(closedEyeDir, fileName))
            yRaw.append(0)

      openEyeDir = os.path.join(kDataPath, "openRightEyes")
      for fileName in os.listdir(openEyeDir):
         if fileName.endswith(".jpg"):
            xRaw.append(os.path.join(openEyeDir, fileName))
            yRaw.append(0)

   if shuffle:
      random.seed(1000)

      index = [i for i in range(len(xRaw))]
      random.shuffle(index)

      xOrig = xRaw
      xRaw = []

      yOrig = yRaw
      yRaw = []

      for i in index:
         xRaw.append(xOrig[i])
         yRaw.append(yOrig[i])

   numberForTesting = int(len(xRaw) * percentForTesting)


   # xTrainRaw, yTrainRaw, xTestRaw, yTestRaw
   return (xRaw[numberForTesting:], yRaw[numberForTesting:], xRaw[:numberForTesting], yRaw[:numberForTesting])
