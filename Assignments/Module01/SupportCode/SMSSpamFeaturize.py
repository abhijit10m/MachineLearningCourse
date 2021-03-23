import math
from collections import defaultdict 

class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use hand crafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures
        
        self.ResetVocabulary()
        
    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw):
        return str.split(xRaw)

    def WordCounts(self, x):
        word_counts = defaultdict(int) 

        # Create a dictionary of words to the count across all samples
        for example in x:
            words = set(self.Tokenize(example))
            for word in words:
                word_counts[word] += 1

        return word_counts

    def SortAllWords(self, x):
        word_counts = self.WordCounts(x) 
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_counts

    def BuildContigencyTables(self, x, y, word_counts):

        contigency_tables = defaultdict(lambda: [[0,0],[0,0]])

        for i in range(len(y)):
            example = x[i]
            example_words = self.Tokenize(example)

            for word in word_counts.keys():
                x_index = 0
                if (word in example_words):
                    x_index = 1
                contigency_tables[word][y[i]][x_index] += 1

        return contigency_tables

    def Probability(self, observed, total_samples):
        # Additive smoothing
        return (observed + 1) / (total_samples + 2)

    def FindTopWordsByMutualInformation(self, x, y, n):
        word_counts = self.WordCounts(x)

        contigency_tables = self.BuildContigencyTables(x, y, word_counts)

        mutual_information = defaultdict(int)

        for word in word_counts.keys():
            c_table = contigency_tables[word]

            sum_c_table = 0
            for i in [0,1]:
                for j in [0, 1]:
                    sum_c_table += c_table[i][j]

            p_x_0 = self.Probability(c_table[0][0] + c_table[1][0], sum_c_table)
            p_x_1 = self.Probability(c_table[0][1] + c_table[1][1], sum_c_table)
            p_y_0 = self.Probability(c_table[0][0] + c_table[0][1], sum_c_table)
            p_y_1 = self.Probability(c_table[1][0] + c_table[1][1], sum_c_table)

            p_y_0_x_0 = self.Probability(c_table[0][0], sum_c_table) 
            p_y_0_x_1 = self.Probability(c_table[0][1], sum_c_table) 
            p_y_1_x_0 = self.Probability(c_table[1][0], sum_c_table) 
            p_y_1_x_1 = self.Probability(c_table[1][1], sum_c_table)

            mi_y_0_x_0 = p_y_0_x_0 * math.log(p_y_0_x_0 / (p_y_0 * p_x_0))
            mi_y_0_x_1 = p_y_0_x_1 * math.log(p_y_0_x_1 / (p_y_0 * p_x_1))
            mi_y_1_x_0 = p_y_1_x_0 * math.log(p_y_1_x_0 / (p_y_1 * p_x_0))
            mi_y_1_x_1 = p_y_1_x_1 * math.log(p_y_1_x_1 / (p_y_1 * p_x_1))
            
            mi = mi_y_0_x_0 + mi_y_0_x_1 + mi_y_1_x_0 + mi_y_1_x_1

            mutual_information[word] = mi
            
        sorted_mutual_information = sorted(mutual_information.items(), key=lambda x: x[1], reverse=True)
        top_n_mutual_information = sorted_mutual_information[:n]

        return top_n_mutual_information

    def FindMostFrequentWords(self, x, n):
        return self.SortAllWords(x)[:n]

    def CreateVocabulary(self, xTrainRaw, yTrainRaw, numFrequentWords=0, numMutualInformationWords=0, supplementalVocabularyWords=[]):
        if self.vocabularyCreated:
            raise UserWarning("Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize.")
            
        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...

        # For now, only use words that are passed in
        self.vocabulary = self.vocabulary + supplementalVocabularyWords
        
        # top n most frequent words:

        if numFrequentWords > 0:
            most_frequent_words = self.FindMostFrequentWords(xTrainRaw, numFrequentWords)
            self.vocabulary += [most_frequent_words[i][0] for i in range(len(most_frequent_words))]

        if numMutualInformationWords > 0:
            top_n_MI_words = self.FindTopWordsByMutualInformation(xTrainRaw, yTrainRaw, numMutualInformationWords)
            self.vocabulary += [top_n_MI_words[i][0] for i in range(len(top_n_MI_words))]

        self.vocabulary = list(dict.fromkeys(self.vocabulary))

        self.vocabularyCreated = True
        
    def _FeaturizeXForVocabulary(self, xRaw): 
        features = []
        
        # for each word in the vocabulary output a 1 if it appears in the SMS string, or a 0 if it does not
        tokens = self.Tokenize(xRaw)
        for word in self.vocabulary:
            if word in tokens:
                features.append(1)
            else:
                features.append(0)
                
        return features

    def _FeaturizeXForHandCraftedFeatures(self, xRaw):
        features = []
        
        # This function can produce an array of hand-crafted features to add on top of the vocabulary related features
        if self.useHandCraftedFeatures:
            # Have a feature for longer texts
            if(len(xRaw)>40):
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if(any(i.isdigit() for i in xRaw)):
                features.append(1)
            else:
                features.append(0)
            
        return features

    def _FeatureizeX(self, xRaw):
        return self._FeaturizeXForVocabulary(xRaw) + self._FeaturizeXForHandCraftedFeatures(xRaw)

    def Featurize(self, xSetRaw):
        return [ self._FeatureizeX(x) for x in xSetRaw ]

    def GetFeatureInfo(self, index):
        if index < len(self.vocabulary):
            return self.vocabulary[index]
        else:
            # return the zero based index of the heuristic feature
            return "Heuristic_%d" % (index - len(self.vocabulary))
