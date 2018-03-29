from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Flatten,Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Embedding, LSTM

from util import *
from iemocap_util import *
# from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
class sentiment:
    def __init__(self):
        self.util = Util()
        self.iemocap_util = util()
    def pre_process(self):
        print("Loading...")
        pos_x, neg_x = self.util.get_data()
        pos_y = [[1,0] for i in pos_x]
        neg_y = [[0,1] for j in neg_x]
        print("Spliting..")
        pos_x_train, pos_x_test, pos_y_train, pos_y_test = train_test_split(pos_x, pos_y, test_size = 0.30, random_state=42)
        neg_x_train, neg_x_test, neg_y_train, neg_y_test = train_test_split(neg_x, neg_y, test_size = 0.30, random_state=42)
        X_train = np.concatenate((pos_x_train, neg_x_train), axis=0)
        Y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)
        X_test = np.concatenate((pos_x_test, neg_x_test), axis=0)
        Y_test = np.concatenate((pos_y_test, neg_y_test), axis=0)
        X_train_encode = [one_hot(d, 2000) for d in X_train]
        X_test_encode = [one_hot(d, 2000) for d in X_test]
        X_train = pad_sequences(X_train_encode, maxlen=200, padding='post')
        X_test = pad_sequences(X_test_encode, maxlen=200, padding='post')
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        X_train, Y_train = shuffle(X_train, Y_train, random_state=10)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=10)
        return X_train, Y_train, X_test, Y_test
    def pre_process_otherData(self):
        print("loading..")
        X_train, Y_train, X_test,Y_test = self.util.get_other_data()
        X_train_encode = [one_hot(d, 2000) for d in X_train]
        X_test_encode = [one_hot(d, 2000) for d in X_test]
        X_train = pad_sequences(X_train_encode, maxlen=250, padding='post')
        X_test = pad_sequences(X_test_encode, maxlen=250, padding='post')
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        # X_train, Y_train = shuffle(X_train, Y_train, random_state=30)
        # X_test, Y_test = shuffle(X_test, Y_test, random_state=30)
        return X_train, Y_train, X_test, Y_test

    def get_iemocap_data(self):
        X, Y = self.iemocap_util.read_iemocap_data()
        X,Y = shuffle(X, Y, random_state=42)
        Y = [self.encode_class(y, ["Positive", "Neutral", "Negative"]) for y in Y]
        X = [one_hot(d, 2000) for d in X]
        X = pad_sequences(X, maxlen=50, padding='post')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=42)
        X_train = np.array(x_train)
        Y_train = np.array(y_train)
        X_test = np.array(x_test)
        Y_test = np.array(y_test)
        return X_train, Y_train, X_test, Y_test
       
    def build_model(self):
        print("Building the model")
        model = Sequential()
        model.add(Embedding(2000, 32, input_length = 50))
        # model.add(LSTM(64 ,return_sequences=True))
        # model.add(Dropout(0.5))
        # model.add(LSTM(128))
        # model.add(Dropout(0.5))
        # model.add(Dense(3, activation='softmax'))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(3))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
        return model
    
    def b_conv(self):
        model = Sequential()
        model.add(Embedding(5000, 32, input_length=500))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    

    def main(self):
        print("preprocessing...")
        X_train, Y_train, X_test, Y_test = self.get_iemocap_data()
        print(X_train.shape, X_test.shape)
        print(Y_train.shape, Y_test.shape)
        model = self.build_model()
        print("Start training..")
        history = model.fit(X_train, Y_train, batch_size = 100, epochs = 10, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        model_json = model.to_json()
        with open("../../models/rnn/sentiment_iemocap2.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("../../models/rnn/sentiment_iemocap2.h5")
        print("Model saved to disk\n")
        print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
        file = open("sentimenet_metrics_iemocap.txt", "w")
        los = "loss:"+str(score[0])
        acc = "accuracy:" + str(score[1])
        file.write(los)
        file.write(acc)
        file.close()
    def encode_class(self,class_name, class_names):
        index = class_names.index(class_name)
        vector = np.zeros(len(class_names))
        vector[index] = 1
        return vector
sentiment = sentiment()
sentiment.main()
