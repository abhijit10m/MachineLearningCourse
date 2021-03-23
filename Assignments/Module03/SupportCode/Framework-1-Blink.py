kOutputDirectory = "/Users/bhatnaa/Documents/uw/csep546/module1/MachineLearningCourse/MachineLearningCourse/visualize/Module3/Assignment1/"

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Learners.AdaBoost as AdaBoost
from joblib import Parallel, delayed
import multiprocessing.dummy as mp


import logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

(xRaw, yRaw) = BlinkDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize

featurizer = BlinkFeaturize.BlinkFeaturize()

# featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, splitGrid3By3=False)

# xTrain    = featurizer.Featurize(xTrainRaw)
# xValidate = featurizer.Featurize(xValidateRaw)
# xTest     = featurizer.Featurize(xTestRaw)

# for i in range(10):
#     print("%d - " % (yTrain[i]), xTrain[i])

sweeps = dict()
sweeps['maxDepth'] = [1,5,10,12,15,18,20,30,35,40]
sweeps['rounds'] = [1,5,10,15,20,25,30,35,40]


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
            yPredict =  model.predict(xValidate, classificationThreshold=threshold)
            FPR = EvaluateBinaryClassification.FalsePositiveRate(yValidate, yPredict)
            FNR = EvaluateBinaryClassification.FalseNegativeRate(yValidate, yPredict)
            logging.debug("yPredict: %s", yPredict)
            logging.debug("yValidate: %s", yValidate)
            logging.debug("threshold : %s, FPR: %s, FNR: %s", threshold, FPR, FNR)
        except NotImplementedError:
            raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

        FPR_dict[threshold] = FPR
        FNR_dict[threshold] = FNR
        return (threshold, FPR, FNR)


    try:
        with mp.Pool() as p:
            p.map(f, thresholds)
        p.close()
        p.join()
        # for t in thresholds:
        #     f(t)

        for threshold in thresholds:
            FPRs.append(FPR_dict[threshold])
            FNRs.append(FNR_dict[threshold])
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)


def plotValidationVsTrainingSetAccuracy(models, sweepName):
    yValues = []
    errorBars = []
    seriesName = ["Training set accuracy","Validation set accuracy"]
    xValues = []
    sweeps[sweepName]

    yValues_train = []
    errorBars_train = []
    yValues_validate = []
    errorBars_validate = []
    min_lowerBound = 1.0
    counter = 0
    for (sweep, model) in models:
        
        # Training Set Accuracy
        yPredict = model.predict(xTrain)
        EvaluateBinaryClassification.ExecuteAll(yTrain, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTrain, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTrain), .5)
        yValues_train.append(accuracy)
        errorBars_train.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        # Validation Set Accuracy
        yPredict = model.predict(xValidate)
        EvaluateBinaryClassification.ExecuteAll(yValidate, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yValidate, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xValidate), .5)
        yValues_validate.append(accuracy)
        errorBars_validate.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        xValues.append(sweeps[sweepName][counter])
        counter += 1
        print("yValues_train %", yValues_train)
        print("yValues_validate %", yValues_validate)
        print("errorBars_train %", errorBars_train)
        print("errorBars_validate %", errorBars_validate)
        print("seriesName %", seriesName)
        print("xValues %", xValues)
        Charting.PlotSeriesWithErrorBars([yValues_train, yValues_validate], [errorBars_train, errorBars_validate], seriesName, xValues, chartTitle="Validation Set vs Training Set accuracy", xAxisTitle=sweepName, yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="ValidationSetAccuracy_{}".format(sweepName))



def plotValidationVsTestingSetAccuracy(models, sweepName):
    yValues = []
    errorBars = []
    seriesName = ["Training set accuracy","Testing set accuracy"]
    xValues = []
    sweeps[sweepName]

    yValues_train = []
    errorBars_train = []
    yValues_test = []
    errorBars_test = []
    min_lowerBound = 1.0
    counter = 0
    for (sweep, model) in models:
        
        # Training Set Accuracy
        yPredict = model.predict(xTrain)
        EvaluateBinaryClassification.ExecuteAll(yTrain, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTrain, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTrain), .5)
        yValues_train.append(accuracy)
        errorBars_train.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        # Validation Set Accuracy
        yPredict = model.predict(xTest)
        EvaluateBinaryClassification.ExecuteAll(yTest, yPredict)
        accuracy = EvaluateBinaryClassification.Accuracy(yTest, yPredict)
        (lowerBound, upperBound) = ErrorBounds.GetAccuracyBounds(accuracy, len(xTest), .5)
        yValues_test.append(accuracy)
        errorBars_test.append(accuracy - lowerBound)
        if lowerBound < min_lowerBound:
            min_lowerBound = lowerBound

        xValues.append(sweeps[sweepName][counter])
        counter += 1
        print("yValues_train %", yValues_train)
        print("yValues_test %", yValues_test)
        print("errorBars_train %", errorBars_train)
        print("errorBars_test %", errorBars_test)
        print("seriesName %", seriesName)
        print("xValues %", xValues)

        Charting.PlotSeriesWithErrorBars([yValues_train, yValues_test], [errorBars_train, errorBars_test], seriesName, xValues, chartTitle="Training Set vs Test Set accuracy", xAxisTitle=sweepName, yAxisTitle="Accuracy", yBotLimit=min_lowerBound - 0.01, outputDirectory=kOutputDirectory, fileName="TestSetAccuracy_{}".format(sweepName))


## Sweep without splitting

# # rounds hyperparamter in BoostedTrees
# models = []
# maxDepth = 5
# adaBoost = AdaBoost.AdaBoost(sweeps['rounds'][-1], DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=maxDepth)
# adaBoostModel = adaBoost.adaBoost(xTrain, yTrain)

# for round in sweeps['rounds']:
#     models.append((round, adaBoostModel.getModelWithRounds(round)))

# models = sorted(models, key = lambda x: x[0])
# plotValidationVsTrainingSetAccuracy(models, "rounds")
# plotValidationVsTestingSetAccuracy(models, "rounds")

## maxDepth for decision trees
# models = []
# def evaluateDecisionTree(maxDepth):
#     adaBoost = AdaBoost.AdaBoost(1, DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=maxDepth)
#     adaBoostModel = adaBoost.adaBoost(xTrain, yTrain)
#     return (maxDepth, adaBoostModel.getModelWithRounds(1))

# models = Parallel(n_jobs=20)(delayed(evaluateDecisionTree)(maxDepth) for maxDepth in sweeps['maxDepth'])

# models = sorted(models, key = lambda x: x[0])
# plotValidationVsTrainingSetAccuracy(models, "maxDepth")
# plotValidationVsTestingSetAccuracy(models, "maxDepth")


    ## Tune rounds first
    ##Sweep Rounds

    # yValues_train = [0.6144544431946006, 0.6144544431946006, 0.6161417322834646, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966]
    # yValues_test = [0.6404494382022472, 0.6404494382022472, 0.6449438202247191, 0.6449438202247191, 0.6449438202247191, 0.6449438202247191, 0.6449438202247191, 0.6449438202247191, 0.6449438202247191]
    # errorBars_train = [0.005468605844865415, 0.005468605844865415, 0.005464113208549004, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581]
    # errorBars_test = [0.01524113140821215, 0.01524113140821215, 0.015198624244549763, 0.015198624244549763, 0.015198624244549763, 0.015198624244549763, 0.015198624244549763, 0.015198624244549763, 0.015198624244549763]
    # seriesName = ['Training set accuracy', 'Testing set accuracy']
    # xValues = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    # min_lowerBound = 0.5



    # yValues_train = [0.6144544431946006, 0.6144544431946006, 0.6161417322834646, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966, 0.6169853768278966]
    # yValues_validate = [0.6112359550561798, 0.6112359550561798, 0.6157303370786517, 0.6179775280898876, 0.6179775280898876, 0.6179775280898876, 0.6179775280898876, 0.6179775280898876, 0.6179775280898876]
    # errorBars_train = [0.005468605844865415, 0.005468605844865415, 0.005464113208549004, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581, 0.005461840829562581]
    # errorBars_validate = [0.015482542820047573, 0.015482542820047573, 0.015449275678294905, 0.015432120075171474, 0.015432120075171474, 0.015432120075171474, 0.015432120075171474, 0.015432120075171474, 0.015432120075171474]
    # seriesName = ['Training set accuracy', 'Validation set accuracy']
    # xValues = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    # min_lowerBound = 0.5



    ## Sweeps Maxdepth
    # yValues_train % [0.6161417322834646, 0.6352643419572553, 0.6850393700787402, 0.8048368953880765, 0.999156355455568, 1.0, 1.0, 1.0, 1.0, 1.0]
    # yValues_validate % [0.6157303370786517, 0.6382022471910113, 0.604494382022472, 0.5932584269662922, 0.5438202247191011, 0.5550561797752809, 0.5775280898876405, 0.5460674157303371, 0.5550561797752809, 0.5528089887640449]
    # errorBars_train % [0.005464113208549004, 0.00540829381967789, 0.0052189102409262444, 0.004452938777933402, 0.000326204843640876, 0.0, 0.0, 0.0, 0.0, 0.0]
    # errorBars_validate % [0.015449275678294905, 0.015261839930557408, 0.015529849969976861, 0.015601847805972069, 0.015819417508758815, 0.015783955869206667, 0.015688458474445777, 0.01581297585301944, 0.015783955869206667, 0.015791699745972942]
    # seriesName % ['Training set accuracy', 'Validation set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]

    # yValues_train % [0.6161417322834646, 0.6352643419572553, 0.6850393700787402, 0.8048368953880765, 0.999156355455568, 1.0, 1.0, 1.0, 1.0, 1.0]
    # yValues_test % [0.6449438202247191, 0.6741573033707865, 0.647191011235955, 0.651685393258427, 0.6157303370786517, 0.5887640449438202, 0.5932584269662922, 0.5752808988764045, 0.6089887640449438, 0.5528089887640449]
    # errorBars_train % [0.005464113208549004, 0.00540829381967789, 0.0052189102409262444, 0.004452938777933402, 0.000326204843640876, 0.0, 0.0, 0.0, 0.0, 0.0]
    # errorBars_test % [0.015198624244549763, 0.014886047988616258, 0.01517682255003372, 0.015132115038664073, 0.015449275678294905, 0.01562827255851129, 0.015601847805972069, 0.015699494570986405, 0.015498656599712501, 0.015791699745972942]
    # seriesName % ['Training set accuracy', 'Testing set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]


    ## Tune maxdepth first
    ## Sweeps Maxdepth

    # yValues_train % [0.6144544431946006, 0.6282339707536558, 0.6616985376827896, 0.6782902137232846, 0.7120359955005624, 0.750281214848144, 0.7859955005624297, 0.9232283464566929, 0.9606299212598425, 0.983970753655793]
    # yValues_validate % [0.6112359550561798, 0.6269662921348315, 0.6247191011235955, 0.6202247191011236, 0.6067415730337079, 0.5955056179775281, 0.597752808988764, 0.5707865168539326, 0.5550561797752809, 0.550561797752809]
    # errorBars_train % [0.005468605844865415, 0.005429870677993165, 0.00531588958216167, 0.005248483524113223, 0.005087612237070882, 0.004863305702780263, 0.004608031176756011, 0.002991224016730798, 0.0021850185237342057, 0.0014110473040949145]
    # errorBars_validate % [0.015482542820047573, 0.015359989962321352, 0.015378551089413661, 0.015414614909093682, 0.015514425299301582, 0.015588128447167504, 0.015574069916391875, 0.01572057142715355, 0.015783955869206667, 0.015799117399508233]
    # seriesName % ['Training set accuracy', 'Validation set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]

    # yValues_train % [0.6144544431946006, 0.6282339707536558, 0.6616985376827896, 0.6782902137232846, 0.7120359955005624, 0.750281214848144, 0.7859955005624297, 0.9232283464566929, 0.9606299212598425, 0.983970753655793]
    # yValues_test % [0.6404494382022472, 0.6651685393258427, 0.6584269662921348, 0.6539325842696629, 0.6314606741573033, 0.6426966292134831, 0.647191011235955, 0.6, 0.6022471910112359, 0.5797752808988764]
    # errorBars_train % [0.005468605844865415, 0.005429870677993165, 0.00531588958216167, 0.005248483524113223, 0.005087612237070882, 0.004863305702780263, 0.004608031176756011, 0.002991224016730798, 0.0021850185237342057, 0.0014110473040949145]
    # errorBars_test % [0.01524113140821215, 0.014989038529603094, 0.01506226868419891, 0.015109205954758953, 0.01532180286259377, 0.015220060015959547, 0.01517682255003372, 0.015559671294295252, 0.015544931635840697, 0.01567708967171444]
    # seriesName % ['Training set accuracy', 'Testing set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]


    ## Sweep Rounds
    # yValues_train % [0.6282339707536558, 0.6366704161979753, 0.6352643419572553, 0.6372328458942632, 0.6375140607424072, 0.6380764904386952, 0.6380764904386952, 0.6377952755905512, 0.6377952755905512]
    # yValues_validate % [0.6269662921348315, 0.6382022471910113, 0.6382022471910113, 0.6382022471910113, 0.6382022471910113, 0.6359550561797753, 0.6359550561797753, 0.6359550561797753, 0.6359550561797753]
    # errorBars_train % [0.005429870677993165, 0.005403829556038242, 0.00540829381967789, 0.005402029881438675, 0.005401127046770204, 0.005399315377683789, 0.005399315377683789, 0.005400222212520234, 0.005400222212520234]
    # errorBars_validate % [0.015359989962321352, 0.015261839930557408, 0.015261839930557408, 0.015261839930557408, 0.015261839930557408, 0.015282187058144325, 0.015282187058144325, 0.015282187058144325, 0.015282187058144325]
    # seriesName % ['Training set accuracy', 'Validation set accuracy']
    # xValues % [1, 5, 10, 15, 20, 25, 30, 35, 40]

    # yValues_train % [0.6282339707536558, 0.6366704161979753, 0.6352643419572553, 0.6372328458942632, 0.6375140607424072, 0.6380764904386952, 0.6380764904386952, 0.6377952755905512, 0.6377952755905512]
    # yValues_test % [0.6651685393258427, 0.6696629213483146, 0.6741573033707865, 0.6764044943820224, 0.6741573033707865, 0.6696629213483146, 0.6696629213483146, 0.6696629213483146, 0.6696629213483146]
    # errorBars_train % [0.005429870677993165, 0.005403829556038242, 0.00540829381967789, 0.005402029881438675, 0.005401127046770204, 0.005399315377683789, 0.005399315377683789, 0.005400222212520234, 0.005400222212520234]
    # errorBars_test % [0.014989038529603094, 0.014938314057465818, 0.014886047988616258, 0.014859331790670116, 0.014886047988616258, 0.014938314057465818, 0.014938314057465818, 0.014938314057465818, 0.014938314057465818]
    # seriesName % ['Training set accuracy', 'Testing set accuracy']
    # xValues % [1, 5, 10, 15, 20, 25, 30, 35, 40]


## With grids

## Tune maxDepth first
    # yValues_train % [0.6706974128233971, 0.8101799775028121, 0.953880764904387, 0.9828458942632171, 0.9943757030371203, 0.9969066366704162, 0.9971878515185602, 0.9971878515185602, 0.9971878515185602, 0.9971878515185602]
    # yValues_validate % [0.6404494382022472, 0.7775280898876404, 0.7820224719101123, 0.7707865168539326, 0.7887640449438202, 0.7887640449438202, 0.7887640449438202, 0.7887640449438202, 0.7887640449438202, 0.7887640449438202]
    # errorBars_train % [0.00528025399087928, 0.004406113597100414, 0.0023565790911714846, 0.0014588837330044013, 0.000840239898967643, 0.0006239311041755791, 0.0005949788663828226, 0.0005949788663828226, 0.0005949788663828226, 0.0005949788663828226]
    # errorBars_validate % [0.01524113140821215, 0.013209623298411133, 0.01311324801802527, 0.013350020485784264, 0.012964395495206227, 0.012964395495206227, 0.012964395495206227, 0.012964395495206227, 0.012964395495206227, 0.012964395495206227]
    # seriesName % ['Training set accuracy', 'Validation set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]

    # yValues_train % [0.6706974128233971, 0.8101799775028121, 0.953880764904387, 0.9828458942632171, 0.9943757030371203, 0.9969066366704162, 0.9971878515185602, 0.9971878515185602, 0.9971878515185602, 0.9971878515185602]
    # yValues_test % [0.698876404494382, 0.7887640449438202, 0.8089887640449438, 0.8067415730337079, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    # errorBars_train % [0.00528025399087928, 0.004406113597100414, 0.0023565790911714846, 0.0014588837330044013, 0.000840239898967643, 0.0006239311041755791, 0.0005949788663828226, 0.0005949788663828226, 0.0005949788663828226, 0.0005949788663828226]
    # errorBars_test % [0.014570262626523833, 0.012964395495206227, 0.012485199878636277, 0.01254097309537805, 0.01270441841215142, 0.01270441841215142, 0.01270441841215142, 0.01270441841215142, 0.01270441841215142, 0.01270441841215142]
    # seriesName % ['Training set accuracy', 'Testing set accuracy']
    # xValues % [1, 5, 10, 12, 15, 18, 20, 30, 35, 40]


## Tune number of rounds

    # yValues_train % [0.8101799775028121, 0.8661417322834646, 0.9007311586051744, 0.9372890888638921, 0.9468503937007874, 0.9668166479190101, 0.9786276715410573, 0.9895950506186727, 0.9949381327334084]
    # yValues_validate % [0.7775280898876404, 0.8112359550561797, 0.8359550561797753, 0.8314606741573034, 0.8337078651685393, 0.8606741573033708, 0.8651685393258427, 0.8651685393258427, 0.8674157303370786]
    # errorBars_train % [0.004406113597100414, 0.003825700862554071, 0.003359680636192741, 0.002723969638549595, 0.002520488002458965, 0.0020124547038198326, 0.0016249073096862299, 0.0011400992755347117, 0.0007973469569437786]
    # errorBars_validate % [0.013209623298411133, 0.012428766535364177, 0.011761634687464606, 0.01188957360870646, 0.01182599254539829, 0.01099841285246439, 0.010847777685454862, 0.010847777685454862, 0.010770960776961047]
    # seriesName % ['Training set accuracy', 'Validation set accuracy']
    # xValues % [1, 5, 10, 15, 20, 25, 30, 35, 40]

    # yValues_train % [0.8101799775028121, 0.8661417322834646, 0.9007311586051744, 0.9372890888638921, 0.9468503937007874, 0.9668166479190101, 0.9786276715410573, 0.9895950506186727, 0.9949381327334084]
    # yValues_test % [0.7887640449438202, 0.8382022471910112, 0.8471910112359551, 0.8629213483146068, 0.8808988764044944, 0.8786516853932584, 0.8786516853932584, 0.8764044943820225, 0.8831460674157303]
    # errorBars_train % [0.004406113597100414, 0.003825700862554071, 0.003359680636192741, 0.002723969638549595, 0.002520488002458965, 0.0020124547038198326, 0.0016249073096862299, 0.0011400992755347117, 0.0007973469569437786]
    # errorBars_test % [0.012964395495206227, 0.011696487212386142, 0.011427728600555631, 0.01092358810638061, 0.010287639908932555, 0.010370985924518128, 0.010370985924518128, 0.010453180135625462, 0.01020311386272621]
    # seriesName % ['Training set accuracy', 'Testing set accuracy']
    # xValues % [1, 5, 10, 15, 20, 25, 30, 35, 40]

maxDepth = 5
rounds = 25

seriesFPRs = []
seriesFNRs = []
seriesLabels = ["Without Grid"]

adaBoost = AdaBoost.AdaBoost(rounds, DecisionTreeWeighted.DecisionTreeWeighted, maxDepth=maxDepth)

featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, splitGrid3By3=False)


xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

adaBoostModelWithoutGrid = adaBoost.adaBoost(xTrain, yTrain).getModelWithRounds(rounds)

(nogridFPRs, nogridgridFNRs, thresholds) = TabulateModelPerformanceForROC(adaBoostModelWithoutGrid, xValidate, yValidate)
seriesFPRs.append(nogridFPRs)
seriesFNRs.append(nogridgridFNRs)
print("adaBoostModelWithoutGrid Test Rate {}".format(list(zip(thresholds, nogridFPRs, nogridgridFNRs))))

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Boosted Tree with vs without grid", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Blink-WithVsWithout-Grid")



featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=True, splitGrid3By3=True)

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

adaBoostModelWithGrid = adaBoost.adaBoost(xTrain, yTrain).getModelWithRounds(rounds)


(gridFPRs, gridFNRs, thresholds) = TabulateModelPerformanceForROC(adaBoostModelWithGrid, xValidate, yValidate)
seriesFPRs.append(gridFPRs)
seriesFNRs.append(gridFNRs)
seriesLabels = ["Without Grid", "With Grid"]
print("adaBoostModelWithGrid Test Rate {}".format(list(zip(thresholds, gridFPRs, gridFNRs))))
Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Boosted Tree with vs without grid", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-Blink-WithVsWithout-Grid")