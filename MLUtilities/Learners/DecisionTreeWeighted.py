import collections
import math
import time
import copy


def __Entropy(y, w, labelDistribution):
    pLogP = []

    for label in labelDistribution:
        weightDistribution = 0
        for i in range(len(y)):
            if y[i] == label:
                weightDistribution += w[i]

        lableProbability = 0

        if sum(w) != 0:
            lableProbability = weightDistribution / sum(w)

        if lableProbability == 0:
            pLogP.append(0)
        else:
            pLogP.append(lableProbability * math.log2(lableProbability))

    entropyValue = -1 * (sum(pLogP))

    return entropyValue

def Entropy(y, w):
    labelDistribution = collections.Counter()
    for i in range(len(y)):
        labelDistribution[y[i]] += w[i]
    return __Entropy(y, w, labelDistribution)


def __FindBestSplitOnFeature(x, y, w,featureIndex, labelDistribution):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return (None, None, None)

    ### Do more tests to make sure you haven't hit a terminal case...

    # order based on the value of the feature at featureIndex
    # so x[indexesInSortedOrder[0]] will be the training sample with the smalles value of 'featureIndex'
    # and y[indexesInSortedOrder[0]] will be the associated label
    if len(w) == 0:
        w = [1.0 for label in y]
        
    indexesInSortedOrder = sorted(range(len(x)), key = lambda i : x[i][featureIndex])
    bestThreshold = float((x[0][featureIndex] + x[0][featureIndex] - 1)/2)
    entropyAfterSplit = Entropy(y, w)
    
    y_sorted = []
    w_sorted = []

    for i in range(len(indexesInSortedOrder)):
        y_sorted.append(y[indexesInSortedOrder[i]])
        w_sorted.append(w[indexesInSortedOrder[i]])

    prevThreshold = bestThreshold
    lastThresholdChangeIndex = 0
    for i in range(1, len(indexesInSortedOrder)):
        prevIndex = indexesInSortedOrder[i - 1]
        index = indexesInSortedOrder[i]

        threshold = float((x[prevIndex][featureIndex] + x[index][featureIndex]) / 2)
        if(threshold > prevThreshold):

            # if y[prevIndex] != y[index]:
            entropy =  Entropy(y_sorted[:i], w_sorted[:i]) * sum(w_sorted[:i])
            entropy +=  Entropy(y_sorted[i:], w_sorted[i:]) * sum(w_sorted[i:])
            entropy = entropy / len(y)

            if entropy < entropyAfterSplit:
                bestThreshold = threshold
                entropyAfterSplit = entropy

        prevThreshold = threshold

    splitLessThanThreshold = [[],[],[]]
    splitGreaterThanEqualToThreshold = [[],[],[]]
    
    for i in indexesInSortedOrder:
        if x[i][featureIndex] < bestThreshold:
            splitLessThanThreshold[0].append(x[i])
            splitLessThanThreshold[1].append(y[i])
            splitLessThanThreshold[2].append(w[i])
        else:
            splitGreaterThanEqualToThreshold[0].append(x[i])
            splitGreaterThanEqualToThreshold[1].append(y[i])
            splitGreaterThanEqualToThreshold[2].append(w[i])

    # HINT: might like to return the partitioned data and the
    #  entropy after partitioning based on the threshold
    splitData = [splitLessThanThreshold, splitGreaterThanEqualToThreshold]
    return (bestThreshold, splitData, entropyAfterSplit)
 
def FindBestSplitOnFeature(x, y, featureIndex, w):
    labelDistribution = collections.Counter()
    for i in range(len(y)):
        labelDistribution[y[i]] += w[i]
    return __FindBestSplitOnFeature(x, y, w, featureIndex, labelDistribution)

def InformationGain(x, y, w, featureIndex):
    (bestThreshold, splitData, splitEntropy) = FindBestSplitOnFeature(x, y, featureIndex, w)

    if (bestThreshold, splitData, splitEntropy) == (None, None, None):
        # very minimal information gain.
        return (-999999999, 0, 0)

    return ((Entropy(y, w) - splitEntropy), bestThreshold, splitData)

def FindBestSplitAttribute(x, y, w):

    featureIndex = 0
    bestThreshold = 0

    informationGains = []

    for featureIndex in range(len(x[0])):
        (informationGain, threshold, splitData) = InformationGain(x, y, w, featureIndex)
        informationGains.append([informationGain, featureIndex, threshold, splitData])

    informationGains = sorted(informationGains, key=lambda gain : gain[0], reverse=True)

    if informationGains[0][0] <= 0:
        # print("Information gain is negative")
        return (None, None, None)

    featureIndex = informationGains[0][1]
    bestThreshold = informationGains[0][2]
    splitData = informationGains[0][3]
    return (featureIndex, bestThreshold, splitData)

class TreeNode(object):
    def __init__(self, depth = 0, split_features_count = 0):
        self.depth = depth
        self.labelDistribution = collections.Counter()
        self.splitIndex = None
        self.threshold = None
        self.children = []
        self.x = []
        self.y = []
        self.w = []
        self.split_features_count = split_features_count

    def isLeaf(self):
        return self.splitIndex == None

    def addData(self, x, y, w=None):
        self.x += x
        self.y += y
        if(w == None):
            w = [ 1.0 for label in y ]
        self.w += w

        for i in range(len(y)):
            self.labelDistribution[y[i]] += 1

    def growTree(self, maxDepth):
        if self.depth == maxDepth:
            # print("Exit condition maxDepth")
            return

        # Condition that all examples are classified under one label.
        if len(self.labelDistribution.most_common()) == 1:
            # print("Exit condition all classified in leaf")
            return
    
        if (len(self.y) == 0):
            # print("Exit condition no more labels to classify")
            return
            
        # bestThreshold could be 0 or 1 (binary values of X)
        # featureIndex could be any index in X
        (featureIndex, bestThreshold, splitData) = FindBestSplitAttribute(self.x, self.y, self.w)

        if (featureIndex, bestThreshold) == (self.splitIndex, self.threshold):
            # print("Exit condition : Trying to split on the same feature again")
            return


        # print("Splitting on : {}, {}".format(featureIndex, bestThreshold))
        if (featureIndex, bestThreshold, splitData) == (None, None, None):
            # print("Exit condition : No information gain by splitting")
            return

        self.splitIndex = featureIndex
        self.threshold = bestThreshold

        x_0 = splitData[0][0]
        y_0 = splitData[0][1]
        w_0 = splitData[0][2]
        x_1 = splitData[1][0]
        y_1 = splitData[1][1]
        w_1 = splitData[1][2]

        x_0_TreeNode = TreeNode(depth=self.depth + 1, split_features_count=self.split_features_count + 1)
        x_0_TreeNode.addData(x_0,y_0,w_0)

        x_1_TreeNode = TreeNode(self.depth + 1, split_features_count=self.split_features_count + 1)
        x_1_TreeNode.addData(x_1,y_1,w_1)

        self.children.append(x_0_TreeNode)
        self.children.append(x_1_TreeNode)

        x_0_TreeNode.growTree(maxDepth)
        x_1_TreeNode.growTree(maxDepth)
        
    def predictProbability(self, example):
        # Remember to find the correct leaf then use an m-estimate to smooth the probability:
        #  (#_with_label_1 + 1) / (#_at_leaf + 2)
        
        if self.isLeaf():
            return (sum([self.w[i] if self.y[i] == 1 else 0 for i in range(len(self.y))]) + 1) / (sum(self.w) + 2)
        else:
            if example[self.splitIndex] < self.threshold:
                return self.children[0].predictProbability(example)
            else:
                return self.children[1].predictProbability(example)
    
    def visualize(self, depth=1):
        ## Here is a helper function to visualize the tree (if you choose to use the framework class)
        if self.isLeaf():
            print(self.labelDistribution)

        else:
            print("Split on index: %d" % (self.splitIndex))

            # less than
            for i in range(depth):
                print(' ', end='', flush=True)
            print("< %f -- " % self.threshold, end='', flush=True)
            self.children[0].visualize(depth+1)

            # greater than or equal
            for i in range(depth):
                print(' ', end='', flush=True)
            print(">= %f -- " % self.threshold, end='', flush=True)
            self.children[1].visualize(depth+1)

    def countNodes(self):
        if self.isLeaf():
            return 1

        else:
            return 1 + self.children[0].countNodes() + self.children[1].countNodes()

class DecisionTreeWeighted(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        pass

    def fit(self, x, y, w = None, maxDepth = 10000, verbose=True):
        self.maxDepth = maxDepth

        if(w == None):
            w = [ 1.0 for label in y ]

        startTime = time.time()

        self.treeNode = TreeNode(depth=0)

        self.treeNode.addData(x,y,w)
        self.treeNode.growTree(maxDepth)
        
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("Weighted Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." % (self.countNodes(), runtime, len(x[0]), maxDepth))
            

    def predictProbabilities(self, x):
        y = []

        for example in x:
            y.append(self.treeNode.predictProbability(example))        
            
        return y

    def predict(self, x, classificationThreshold=0.5):
        return [ 1 if probability >= classificationThreshold else 0 for probability in self.predictProbabilities(x) ]

    def visualize(self):
        self.treeNode.visualize()

    def countNodes(self):
        return self.treeNode.countNodes()