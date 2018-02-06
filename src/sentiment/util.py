from os import listdir
from os.path import isfile, join
import re
import io
import numpy as np
class Util:
    def __init__(self,path="../../datasets/corpus/sentiment/movie_review/"):
        self.path = path
        self.path2 = "../../datasets/corpus/sentiment/aclImdb/"
    def get_data(self):
        positiveFiles = [self.path + 'positiveReviews/' + f for f in
        listdir(self.path +'positiveReviews/') if isfile(join(self.path +'positiveReviews/', f))]

        negativeFiles = [self.path + 'negativeReviews/' + f for f in
        listdir(self.path +'negativeReviews/') if isfile(join(self.path +'negativeReviews/', f))]

        return self.get_sentence(positiveFiles, negativeFiles)

    def get_sentence(self, positiveFiles, negativeFiles):
        pos_sentence = []
        neg_sentence = []

        for pf in positiveFiles:
            with open(pf, mode="r") as f:
                line = f.readline()
                pos_sentence.append(self.cleanSentences(line))
        
        for nf in negativeFiles:
            with open(nf, mode="r") as f:
                line = f.readline()
                neg_sentence.append(self.cleanSentences(line))
        return pos_sentence, neg_sentence
    def get_other_data(self):
        pos_x_train = []
        pos_y_train = []
        pos_x_test = []
        pos_y_test = []
        
        neg_x_train = []
        neg_y_train = []
        neg_x_test = []
        neg_y_test = []
        for f in [file for file in listdir(self.path2 + "train/pos/")]:
            with open(self.path2 + "train/pos/" + f, mode="r") as fi:
                pos_x_train.append(self.cleanSentences(fi.readline()))
                pos_y_train.append([1,0])
        for f in [file for file in listdir(self.path2 + "train/neg/")]:
            with open(self.path2 + "train/neg/" + f, mode="r") as fi:
                neg_x_train.append(self.cleanSentences(fi.readline()))
                neg_y_train.append([0,1])

        for f in [file for file in listdir(self.path2 + "test/pos/")]:
            with open(self.path2 + "test/pos/" + f, mode="r") as fi:
                pos_x_test.append(self.cleanSentences(fi.readline()))
                pos_y_test.append([1,0])
        for f in [file for file in listdir(self.path2 + "test/neg/")]:
            with open(self.path2 + "test/neg/" + f, mode="r") as fi:
                neg_x_test.append(self.cleanSentences(fi.readline()))
                neg_y_test.append([0,1])
        X_train = np.concatenate((pos_x_train, neg_x_train), axis=0)
        Y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)
        X_test = np.concatenate((pos_x_test, neg_x_test), axis=0)
        Y_test = np.concatenate((pos_y_test, neg_y_test), axis=0)
        return X_train, Y_train, X_test,Y_test
    
        


    def cleanSentences(self,string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())      
# u = Util()
# u.get_other_data()