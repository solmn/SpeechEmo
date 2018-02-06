import os
import csv
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Activation
from keras.models import Sequential
from  scipy.stats import signaltonoise
import keras.optimizers 
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
x, y = get_sample("../../datasets/features/augmented_mfcc/")
x = sequence.pad_sequences(x, maxlen=200, padding='post', truncating='post')
x = (x - np.mean(x))/np.std(x)
print("xx",x.shape)
emotions = ["angry", "fear", "happy", "neutral", "sad","surprise"]
# emotions = ["neutral", "positive"]
Y = []
# for i, label in enumerate(y):
#     Y.append(encode_class(label, emotions))
y = label_binarize(y, emotions)
# y = np.array(Y, dtype=int)
print(y[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
# x_train = sequence.pad_sequences(x_train, maxlen=200, padding='post', truncating='post')
# x_test = sequence.pad_sequences(x_test,maxlen=200, padding='post', truncating='post')
print("x_t",x_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(x.shape[1], 13)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))
model.add(Dropout(0.5))
model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=70,
          epochs=70,
          validation_data=[x_test, y_test])
score = model.evaluate(x_test, y_test, verbose=0)
print("Model has finished. Accuracy:" + str(score[1]) + " and loss:" + str(score[0]))
model_json = model.to_json()
with open("../../models/rnn/sequence_model_all_agumented.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("../../models/rnn/sequence_weight_all_agumented.h5")
    print("Model saved to disk\n")

