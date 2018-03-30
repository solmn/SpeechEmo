import numpy as np
from keras.layers import *
from keras.models import *
import os
import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import Model
from sklearn.utils import shuffle

input_path = "../../datasets/features/cnn_lstm_features/"
def load_features(input_path =""):
    
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    class_names = os.listdir(input_path)
    i = 0
    for class_name in class_names:
        train_test_path = input_path  + class_name + "/"
        train = train_test_path + "train/"
        test = train_test_path + "test/"
        x_train = np.load(train + "_X.npy")
        x_test = np.load(test + "_X.npy")
        y_train = np.load(train + "_Y.npy")
        y_test = np.load(test + "_Y.npy")
        # print( class_name + ":x_train: " + str(x_train.shape))
        # print( class_name + ":y_train " + str(y_train.shape))
        # print( class_name + ":x_test: " + str(x_test.shape))
        # print( class_name + ":y_test " + str(y_test.shape))
        if(i ==0):
            X_train, Y_train, X_test, Y_test = x_train, y_train, x_test, y_test
        else:
            X_train = np.concatenate((X_train, x_train))
            Y_train = np.concatenate((Y_train, y_train))
            X_test = np.concatenate((X_test, x_test))
            Y_test = np.concatenate((Y_test, y_test))
        i = i + 1
    print("------------------ ", class_name, " -------------------")
    print("X_train: " + str(X_train.shape))
    print("Y_train: " + str(Y_train.shape))
    print("X_test: " + str(X_test.shape))
    print("Y_test: " + str(Y_test.shape))
    # print('-----------------------------------------------------------\n\n')
    # print("ss",X_train.shape[0])
    # print(Y_train[0].shape)
    # print(Y_train[0][0])
    # print(Y_train[10][0])
    return X_train, Y_train, X_test, Y_test

def b_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    inpu_s = (None, X_train[0].shape[1], X_train[0].shape[2], X_train[0].shape[3])
    model.add(TimeDistributed(Conv2D(64, kernel_size=(6,6), padding="valid", activation="relu"), input_shape=inpu_s))
    model.add(TimeDistributed(Conv2D(64, kernel_size=(6,6), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3),  padding="same")))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(128, kernel_size=(6,6), activation="relu", padding="same")))
    model.add(TimeDistributed(Conv2D(128, kernel_size=(6,6), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3),  padding="same")))
    model.add(TimeDistributed(Dropout(0.2)))

    # model.add(TimeDistributed(Conv2D(256, kernel_size=(6,6), activation="relu", padding="same")))
    # model.add(TimeDistributed(Conv2D(256, kernel_size=(6,6), activation="relu", padding="same")))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3),  padding="same")))
    # model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024, activation="relu")))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
    model.summary()
    return model
def build_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    inpu_s = (None, X_train[0].shape[1], X_train[0].shape[2], X_train[0].shape[3])
    print(inpu_s)
    model.add(TimeDistributed(Conv2D(32, kernel_size=(6,6), padding="same", activation="relu"), input_shape=inpu_s))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3),  padding="same")))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv2D(64, kernel_size=(6,6), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding="same")))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Conv2D(64, kernel_size=(6,6), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding="same")))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024, activation="relu")))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Dropout(0.27))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
    model.summary()
    return model
def custom_generate(X, Y):
    while True:
        for i in range(X.shape[0]):
            l = X[i].shape[0]
            x = X[i].reshape(-1, l, 60, 41, 2)
            y = Y[i][0].reshape(1,4)
            # print(y)
            yield (x, y)

def normalize(X):
    max_1 = []
    min_1 = []
    max_2 = []
    min_2 = []
    for i, x in enumerate(X):
        first_channel = x[:,:,:,0]
        second_channel = x[:,:,:,1]
        max_1.append(np.max(first_channel))
        min_1.append(np.min(first_channel))
        max_2.append(np.max(second_channel))
        min_2.append(np.min(second_channel))

    max_1 = np.max(np.array(max_1))
    max_2 = np.max(np.array(max_2))
    min_1 = np.min(np.array(min_1))
    min_2 = np.min(np.array(min_2))

    
    print("first chanel max:", max_1)
    print("first channel min:", min_1)
    print("second channel max:", max_2)
    print("second channel min:", min_2)
    for i, x in enumerate(X):
        X[i][:,:,:,[0]] = (X[i][:,:,:,[0]]  - max_1 )/ (max_1 - min_1)
        X[i][:,:,:,[1]] = (X[i][:,:,:,[1]]  - max_2 )/ (max_2 - min_2)
    return X
    

def main():
    print("Loading features...")
    X_train, Y_train, X_test, Y_test = load_features(input_path=input_path)
    print("Normalizing features...")
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    X_train, Y_train = shuffle(X_train, Y_train, random_state=40)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=40)
    print("Start training..")
    model = build_model(X_train, Y_train, X_test, Y_test)
    batch_size = 1
    nb_epoch = 25
    model.fit_generator(custom_generate(X_train, Y_train),verbose=1, steps_per_epoch = X_train.shape[0], epochs=nb_epoch,
    validation_data= custom_generate(X_test, Y_test), validation_steps=X_test.shape[0])
    # history = model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,
    #               validation_data=(X_test, Y_test))
    score = model.evaluate_generator(custom_generate(X_test, Y_test), steps=X_test.shape[0],use_multiprocessing=False)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.to_json()
    with open("../../models/cnn/cnn_lstm__3_.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../../models/cnn/cnn_lstm__3_.h5")
    file = open("../../models/cnn/cnn_lstm___3_.txt", "w")
    los = "loss:"+str(score[0])
    acc = "accuracy:" + str(score[1])
    file.write(los)
    file.write(acc)
    file.close()
main()