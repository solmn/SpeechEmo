import numpy as np
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import librosa
import os
import csv
def read_audio(filename):
    (sample_rate, signal) = wav.read(filename)
    return signal, sample_rate
def encode_class(class_name, class_names):
    index = class_names.index(class_name)
    vector = np.zeros(len(class_names))
    vector[index] = 1
    return vector
def save_samples(x, y,path):
    print("coo",path)
    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for i in range(x.shape[0]):
            row = x[i, :].tolist()
            row.append(y[i])
            w.writerow(row)
def extract_features(input_path = "", output_path = ""):
    if not os.path.exists( output_path):
        os.mkdir(output_path, 0755)
    class_names = os.listdir(input_path)
    for class_name in class_names:
        for train_test in (os.listdir(input_path + class_name)):
            path_to_audio = input_path + class_name + "/" + train_test + "/"
            files = os.listdir(path_to_audio)
            files= [f for f in files if f.endswith('.wav')]
            for i, file in enumerate(files):
                audio_path = path_to_audio + file
                sample, sample_rate = read_audio(audio_path)
                features = mfcc(signal = sample, winlen=0.064, winstep=0.032, nfft=1024)
                X = []
                Y = []
                for feat in features:
                    X.append(feat)
                    Y.append(class_name)
                X = np.array(X)
                Y = np.array(Y)
                print(class_name + "/" + train_test + "/" + file)
                # if not os.path.exists( output_path + class_name):
                #    os.mkdir(output_path + class_name, 0755)
                if not os.path.exists( output_path + "/" + train_test):
                   os.mkdir(output_path + "/" + train_test, 0755)
                save_samples(X, Y, output_path+ "/"+train_test + "/" + file[:-4] + ".csv")
                print(file, "  -------------  ",X.shape)
                print(Y.shape)
    # folders = os.listdir(input_path)
    # for folder in folders:
    #     files = os.listdir(input_path + folder)
    #     files = [f for f in files if f.endswith('.wav')]
    #     for i, file in enumerate(files):
    #         audio_path = input_path + folder  + "/" + file
    #         sample, sample_rate = read_audio(audio_path)
    #         features = mfcc(signal = sample, winlen=0.064, winstep=0.032, nfft=1024)
    #         X = []
    #         Y = []
    #         for feat in features:
    #             X.append(feat)
    #             Y.append(folder)
    #         X = np.array(X)
    #         Y = np.array(Y)
    #         save_samples(X, Y, output_path + file[:-4] + ".csv")
    #         # print(Y)
    #         print(file, "  -------------  ",X.shape)
    #         print(Y.shape)
extract_features(input_path = "../../datasets/corpus/IEMOCAP_angry_sad_excited_neutral_test_train/", output_path = "../../datasets/features/new_features/")

            
