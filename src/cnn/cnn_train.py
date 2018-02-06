import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adagrad, SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from keras.regularizers import l2
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import csv
from keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.optimizers import *

def prepare_data( feature_folder = "../../datasets/features/cnn/cnn_augumented/"):
    X_train = np.array([])
    Y_train = np.array([])
    X_test = np.array([])
    Y_test = np.array([])
    class_names = os.listdir(feature_folder)
    for i , class_name in enumerate( class_names ):
        feat_file = os.path.join(feature_folder,class_name, class_name + '_x.npy')
        label_file = os.path.join(feature_folder,class_name, class_name + '_y.npy')
        x_features = np.load(feat_file)
        y_labels = np.load(label_file)
        print( class_name + ": x: " + str(x_features.shape))
        print( class_name + ": y: " + str(y_labels.shape))
        x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, test_size = 0.2, random_state = 42)
        if (i == 0):
            X_train, Y_train, X_test, Y_test = x_train, y_train, x_test, y_test
        else:
            X_train = np.concatenate((X_train, x_train))
            Y_train = np.concatenate((Y_train, y_train))
            X_test = np.concatenate((X_test, x_test))
            Y_test = np.concatenate((Y_test, y_test))
    print("X_train: " + str(X_train.shape))
    print("Y_train: " + str(Y_train.shape))
    print("X_test: " + str(X_test.shape))
    print("Y_test: " + str(Y_test.shape))
    return X_train, Y_train, X_test, Y_test
def build_Vgg_model(X_train, Y_train, X_test, Y_test):
    # input: 60x41 data frames with 2 channels => (60, 41, 2) tensors
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=input_shape,kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64, (2, 2), activation='relu',padding='same',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2), activation='relu',padding='same',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), activation='relu',padding='same',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (2, 2), activation='relu',padding='same',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (2, 2), activation='relu',padding='same',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.2))

    
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(Y_train.shape[1]))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    
    return model
def build_model(X_train, Y_train, X_test, Y_test):
    # input: 60x41 data frames with 2 channels => (60, 41, 2) tensors
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_shape))

    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(Y_train.shape[1]))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,save_name="cm"):
    fig = plt.figure()
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
    fig.savefig(save_name+".png")
def main():
    print("Loading features....")
    X_train, Y_train, X_test, Y_test = prepare_data()
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=42)

    model = build_model(X_train, Y_train, X_test, Y_test)
    # model = build_Vgg_model(X_train, Y_train,X_test, Y_test)
    # model.summary()
    print("Training Model...")
    batch_size = 200
    nb_epoch = 100
    history = model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  verbose=1,
                  validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.to_json()
    with open("../../models/cnn/cnn_au.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../../models/cnn/cnn_au.h5")
    file = open("../../models/cnn/cnn_au.txt", "w")
    los = "loss:"+str(score[0])
    acc = "accuracy:" + str(score[1])
    file.write(los)
    file.write(acc)
    file.close()
    prediction = model.predict(X_test)
    y_pred = prediction.argmax(1)
    cm = confusion_matrix(Y_test, y_pred)
    classes = ['fear', 'surprise', 'neutral', 'angry', 'sad', 'happy']
    plot_confusion_matrix(cm, classes=classes, normalize=False,
						  title='Confusion matrix(test)',save_name="cnn_speech_emotpy_test")
    plt.show()
    # assert len(X_train) == len(Y_train)
    # q = np.random.permutation(len(X_train))
    # X_train = X_train[q]
    # Y_train = Y_train[q]

    # assert len(X_test) == len(Y_test)
    # p = np.random.permutation(len(X_test))
    # X_test = X_test[p]
    # Y_test = Y_test[p]
    # print("Done")
    # print("Building model...")
    # accuracy = []
    # loss = []
    # samples = []
    # tr_loss = []
    # tr_acc = []
    # for i in range(len(X_train)/1000 ):
    #     print ("Modeling: " + str(i) + " of " + str(len(X_train)/1000))
    #     x_t = X_train[:(i +1) * 1000]
    #     y_t = Y_train[:(i + 1) *1000]
    #     input_shape = (x_t.shape[1], x_t.shape[2], x_t.shape[3])
    #     model = Sequential()
    #     model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_shape))

    #     model.add(Conv2D(64, (4, 4), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))

    #     model.add(Conv2D(64, (4, 4), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))

    #     model.add(Conv2D(64, (4, 4), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.2))

    #     model.add(Flatten())
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dropout(0.2))

    #     model.add(Dense(Y_train.shape[1]))
    #     model.add(Activation('softmax'))
    #     model.compile(loss='categorical_crossentropy',
    #             optimizer='adadelta',
    #             metrics=['accuracy'])
    #     print("Training Model...")
    #     batch_size = 120
    #     nb_epoch = 10
    #     history = model.fit(x_t, y_t,
    #                 batch_size=batch_size,
    #                 epochs=nb_epoch,
    #                 verbose=1,
    #                 validation_data=(X_test, Y_test))
    #     score = model.evaluate(X_test, Y_test, verbose=0)
    #     accuracy.append(score[1])
    #     loss.append(score[0])
    #     samples.append(x_t.shape[0])
    #     tr_acc.append(history.history['acc'][-1])
    #     tr_loss.append(history.history['loss'][-1])
    # f = open("cnn_test_accuracy.txt", "w+")
    # for x in accuracy:
    #     f.write(str(x) + ",")
    # f.close()
    # f = open("cnn_train_accuracy.txt", "w+")
    # for x in tr_acc:
    #     f.write(str(x) + ",")
    # f.close()

    # f = open("cnn_train_loss.txt", "w+")
    # for x in tr_loss:
    #     f.write(str(x) + ",")
    # f.close()

    # f = open("cnn_test_loss.txt", "w+")
    # for x in loss:
    #     f.write(str(x) + ",")
    # f.close()
    # plt.plot(accuracy)
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plt.plot(loss)
    # plt.show()
    # with open('cnn_accuracy.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(accuracy)
    # with open('cnn_loss.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(loss)
    # with open('cnn_samples.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(samples)
    # model_json = model.to_json()
    # with open("model/cnn_latest_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model/cnn5_latest_model.h5")
    # print("Model saved to disk\n")
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    # features = np.concatenate((X_train, X_test))
    # labels = np.concatenate((Y_train, Y_test))
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    
main()



