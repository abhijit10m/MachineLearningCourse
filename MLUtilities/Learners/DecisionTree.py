import collections
import math
import time
import copy

def Entropy(labelDistribution):
    pLogP = []

    for label in labelDistribution:

        lableProbability = labelDistribution[label] / sum(labelDistribution.values())

        if lableProbability == 0:
            pLogP.append(0)
        else:
            # print(lableProbability)
            pLogP.append(lableProbability * math.log2(lableProbability))

    return -1 * (sum(pLogP))

def FindBestSplitOnFeature(x, y, featureIndex, labelDistribution):
    if len(y) < 2:
        # there aren't enough samples so there is no split to make
        return (None, None, None)

    ### Do more tests to make sure you haven't hit a terminal case...

    # order based on the value of the feature at featureIndex
    # so x[indexesInSortedOrder[0]] will be the training sample with the smalles value of 'featureIndex'
    # and y[indexesInSortedOrder[0]] will be the associated label

    indexesInSortedOrder = sorted(range(len(x)), key = lambda i : x[i][featureIndex])

    bestThreshold = float((x[0][featureIndex] + x[0][featureIndex] - 1)/2)
    entropyAfterSplit = Entropy(labelDistribution)
    
    greaterThanThresholdLabelDistribution = labelDistribution
    
    lessThanThresholdLabelDistribution = collections.Counter()

    prevThreshold = bestThreshold



    for i in range(1, len(indexesInSortedOrder)):
        prevIndex = indexesInSortedOrder[i - 1]
        index = indexesInSortedOrder[i]

        threshold = float((x[prevIndex][featureIndex] + x[index][featureIndex]) / 2)

        if(threshold > prevThreshold):
            for j in range(i+1):
                if x[indexesInSortedOrder[j]][featureIndex] < threshold and x[indexesInSortedOrder[j]][featureIndex] >= prevThreshold:
                    greaterThanThresholdLabelDistribution[y[indexesInSortedOrder[j]]] -= 1
                    lessThanThresholdLabelDistribution[y[indexesInSortedOrder[j]]] += 1


        entropy =  Entropy(greaterThanThresholdLabelDistribution) * sum(greaterThanThresholdLabelDistribution.values())
        entropy +=  Entropy(lessThanThresholdLabelDistribution) * sum(lessThanThresholdLabelDistribution.values())
        entropy = entropy / len(y)

        if entropy < entropyAfterSplit:
            bestThreshold = threshold
            entropyAfterSplit = entropy

        prevThreshold = threshold
        
    splitLessThanThreshold = [[],[]]
    splitGreaterThanEqualToThreshold = [[],[]]
    
    for i in indexesInSortedOrder:
        if x[i][featureIndex] < bestThreshold:
            splitLessThanThreshold[0].append(x[i])
            splitLessThanThreshold[1].append(y[i])
        else:
            splitGreaterThanEqualToThreshold[0].append(x[i])
            splitGreaterThanEqualToThreshold[1].append(y[i])

    # HINT: might like to return the partitioned data and the
    #  entropy after partitioning based on the threshold
    splitData = [splitLessThanThreshold, splitGreaterThanEqualToThreshold]
    return (bestThreshold, splitData, entropyAfterSplit)


def InformationGain(x, y, featureIndex, labelDistribution):
    (bestThreshold, splitData, splitEntropy) = FindBestSplitOnFeature(x, y, featureIndex, copy.deepcopy(labelDistribution))

    if (bestThreshold, splitData, splitEntropy) == (None, None, None):
        # very minimal information gain.
        return (-999999999, 0, 0)

    return ((Entropy(labelDistribution) - splitEntropy), bestThreshold, splitData)

def FindBestSplitAttribute(x, y, labelDistribution):

    featureIndex = 0
    bestThreshold = 0

    informationGains = []

    for featureIndex in range(len(x[0])):
        (informationGain, threshold, splitData) = InformationGain(x, y, featureIndex, copy.deepcopy(labelDistribution))
        informationGains.append([informationGain, featureIndex, threshold, splitData])

    informationGains = sorted(informationGains, key=lambda gain : gain[0], reverse=True)

    if informationGains[0][0] <= 0:
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
        self.split_features_count = split_features_count

    def isLeaf(self):
        return self.splitIndex == None

    def addData(self, x, y):
        self.x += x
        self.y += y

        for label in y:
            self.labelDistribution[label] += 1

    def growTree(self, maxDepth):
        if self.depth == maxDepth:
            return

        for labelCount in self.labelDistribution:
            if labelCount == len(self.y):
                return
    
        if (len(self.y) == 0):
            return
            
        # bestThreshold could be 0 or 1 (binary values of X)
        # featureIndex could be any index in X
        (featureIndex, bestThreshold, splitData) = FindBestSplitAttribute(self.x, self.y, self.labelDistribution)

        if (featureIndex, bestThreshold, splitData) == (None, None, None):
            return

        self.splitIndex = featureIndex
        self.threshold = bestThreshold

        x_0 = splitData[0][0]
        y_0 = splitData[0][1]
        x_1 = splitData[1][0]
        y_1 = splitData[1][1]

        x_0_TreeNode = TreeNode(depth=self.depth + 1, split_features_count=self.split_features_count + 1)
        x_0_TreeNode.addData(x_0,y_0)

        x_1_TreeNode = TreeNode(self.depth + 1, split_features_count=self.split_features_count + 1)
        x_1_TreeNode.addData(x_1,y_1)

        self.children.append(x_0_TreeNode)
        self.children.append(x_1_TreeNode)

        x_0_TreeNode.growTree(maxDepth)
        x_1_TreeNode.growTree(maxDepth)
        
    def predictProbability(self, example):
        # Remember to find the correct leaf then use an m-estimate to smooth the probability:
        #  (#_with_label_1 + 1) / (#_at_leaf + 2)
        
        if self.isLeaf():
            return (self.labelDistribution[1] + 1) / (sum(self.labelDistribution.values()) + 2)

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

class DecisionTree(object):
    """Wrapper class for decision tree learning."""

    def __init__(self):
        pass

    def fit(self, x, y, maxDepth = 10000, verbose=True):
        self.maxDepth = maxDepth
        
        startTime = time.time()

        self.treeNode = TreeNode(depth=0)

        self.treeNode.addData(x,y)
        self.treeNode.growTree(maxDepth)
        
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("Decision Tree completed with %d nodes (%.2f seconds) -- %d features. Hyperparameters: maxDepth=%d." % (self.countNodes(), runtime, len(x[0]), maxDepth))
            

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
