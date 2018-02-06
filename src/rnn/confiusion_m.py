import os
import csv
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Activation
from keras.models import Sequential
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import librosa
import os
import csv
from keras.preprocessing import sequence
from keras.models import model_from_json
print(__doc__)
from  scipy.stats import signaltonoise
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
def load_model():
    graph = "../../models/rnn/rnn_sequence/sequence_model_all_agumented.json"
    weight = "../../models/rnn/rnn_sequence/sequence_weight_all_agumented.h5"
    json_file = open(graph, 'r')
    loaded_model_json = json_file.read()
    json_file.close
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight)
    return loaded_model
def encode_class(class_name, class_names):
    index = class_names.index(class_name)
    vector = np.zeros(len(class_names))
    vector[index] = 1
    return vector
def get_sample(input_path):
    print("loading features...")
    names = os.listdir(input_path)
    print(input_path)
    tx = []
    ty = []
    i = 0;
    for file in names:
        print(file)
        print("\n")
        x, y = load_sample(input_path,file)
        tx.append(np.array(x, dtype=float))
        ty.append(y[0])
        i +=1
        print(str(len(names)) + " of " + str(i))
    tx = np.array(tx)
    ty = np.array(ty)
    return tx, ty
    
def load_sample(input_path,name):
   
    with open(input_path + name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=",")
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

x, y = get_sample("../../datasets/features/augmented_mfcc/")
x = sequence.pad_sequences(x, maxlen=200, padding='post', truncating='post')
# x = (x - np.mean(x))/np.std(x)
print(x.shape)
x = signaltonoise(x,axis=None)
print(x.shape)
emotions = ["angry", "fear", "happy", "neutral", "sad","surprise"]
# emotions = ["neutral", "positive"]

# for i, label in enumerate(y):
#     Y.append(encode_class(label, emotions))
y = label_binarize(y, emotions)

