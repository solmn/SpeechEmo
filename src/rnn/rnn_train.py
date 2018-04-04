import os
import csv
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Activation
from keras.models import Sequential
from  scipy.stats import signaltonoise
from keras.optimizers import SGD, Adam 
from sklearn.utils import shuffle
def encode_class(class_name, class_names):
    index = class_names.index(class_name)
    vector = np.zeros(len(class_names))
    vector[index] = 1
    return vector
def get_sample(input_path):
    print("loading features...")
    names = os.listdir(input_path)
    tx = []
    ty = []
    for file in names:
        x, y = load_sample(input_path,file)
        tx.append(np.array(x, dtype=float))
        ty.append(y[0])
    tx = np.array(tx)
    ty = np.array(ty)
    tx = sequence.pad_sequences(tx, maxlen=150, padding='post', truncating='post')
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
X_train, Y_train = get_sample("../../datasets/features/new_features/train/")
X_test, Y_test = get_sample("../../datasets/features/new_features/test/")
# x = sequence.pad_sequences(x, maxlen=200, padding='post', truncating='post')
# x = (x - np.mean(x))/np.std(x)
# print("xx",x.shape)
# emotions = ["angry", "fear", "happy", "neutral", "sad","surprise"]
emotions = os.listdir("../../datasets/corpus/IEMOCAP_angry_sad_excited_neutral_test_train/")
print(emotions)
# emotions = ["neutral", "positive"]
Y = []
# for i, label in enumerate(y):
#     Y.append(encode_class(label, emotions))
Y_train = label_binarize(Y_train, emotions)
Y_test = label_binarize(Y_test, emotions)
# y = np.array(Y, dtype=int)
print(Y_train[0])
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
X_test, Y_test = shuffle(X_test, Y_test, random_state=42)
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test - np.mean(X_test))/np.std(X_test)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
# x_train = sequence.pad_sequences(x_train, maxlen=200, padding='post', truncating='post')
# x_test = sequence.pad_sequences(x_test,maxlen=200, padding='post', truncating='post')
print("X_train", X_train.shape)
print("Y_train", Y_train.shape)
print("X_test", X_test.shape)
print("Y_test", Y_test.shape)
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(X_train.shape[1], 13)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))
model.add(Dropout(0.5))
model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.compile(Adam(lr = 1e-3), 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train,
          batch_size=100,
          epochs=50,
          validation_data=[X_test, Y_test])
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
model_json = model.to_json()
with open("../../models/rnn/sequence_model_new_data2.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("../../models/rnn/sequence_weight_new_data2.h5")
    print("Model saved to disk\n")
file = open("../../models/rnn/sequence_new2.txt", "w")
los = "loss:"+str(score[0])
acc = "accuracy:" + str(score[1])
file.write(los)
file.write("\n")
file.write(acc)
file.close()

