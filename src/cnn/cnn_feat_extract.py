import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as np
import numpy as np

def get_class_names ( path="cnn_d"):
    class_names = os.listdir(path)
    print(class_names)
    return class_names

def encode_class(class_name, class_names):
    index = class_names.index(class_name)
    vector = np.zeros(len(class_names))
    vector[index] = 1
    return vector

def windows( data, window_size ):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size /2)
def extract_features( input_path, class_name, train_test, class_names, bands, frames ):
    window_size = 512 * (frames -1)
    log_spectograms = []
    labels = []
    class_files = os.listdir(input_path + class_name + "/" + train_test)
    n_files = len(class_files)
    for i, aud_filename in enumerate( class_files ):
        audio_path = input_path + class_name + "/" +train_test +"/" + aud_filename
        print ("Preprocessing: " + class_name +"_"+train_test +": " + str(i) +  " of " + str(n_files) + " :" + class_name + "/"+train_test +"/"+ aud_filename)
        audio_clip, sr = librosa.load(audio_path)
        for ( start, end ) in windows( audio_clip, window_size ):
            if ( len(audio_clip[start:end]) == int(window_size) ):
                audio_signal = audio_clip[start:end]
                mel_spec = librosa.feature.melspectrogram(audio_signal, n_mels = bands)
                log_spec = librosa.logamplitude(mel_spec)
                log_spec = log_spec.T.flatten()[:, np.newaxis].T
                log_spectograms.append(log_spec)
                labels.append(encode_class(class_name, class_names))
    log_specgrams = np.asarray(log_spectograms).reshape(len(log_spectograms), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return np.array(features), np.array(labels)
def extract_features_mfcc( input_path, class_name, train_test, class_names, bands, frames ):
    window_size = 512 * (frames -1)
    log_spectograms = []
    labels = []
    class_files = os.listdir(input_path + class_name + "/" + train_test)
    n_files = len(class_files)
    for i, aud_filename in enumerate( class_files ):
        audio_path = input_path + class_name + "/" +train_test +"/" + aud_filename
        print ("Preprocessing: " + class_name +"_"+train_test +": " + str(i) +  " of " + str(n_files) + " :" + class_name + "/"+train_test +"/"+ aud_filename)
        audio_clip, sr = librosa.load(audio_path)
        for ( start, end ) in windows( audio_clip, window_size ):
            if ( len(audio_clip[start:end]) == int(window_size) ):
                audio_signal = audio_clip[start:end]
                mel_spec = librosa.feature.melspectrogram(audio_signal, n_mels = bands)
                log_spec = librosa.logamplitude(mel_spec)
                log_spec = log_spec.T.flatten()[:, np.newaxis].T
                log_spectograms.append(log_spec)
                labels.append(encode_class(class_name, class_names))
                
    log_specgrams = np.asarray(log_spectograms).reshape(len(log_spectograms), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return np.array(features), np.array(labels)

def preprocess ( input_path = "cnn_d/", output_path = "cnn_f/" ):
    bands = 60
    frames = 41
    if not os.path.exists( output_path ):
        os.mkdir( output_path, 0755 )
    class_names = get_class_names(path = input_path)
    print("Preprocessing...")
    for i , class_name in enumerate( class_names ):
        if not os.path.exists( output_path + class_name ):
            os.mkdir(output_path + class_name, 0755)
        features, labels = extract_features( input_path, class_name, class_names, bands, frames )
        print ("Features of " + class_name + ":", features.shape)
        print ("Labels of " + class_name + ":", labels.shape)
        feature_file = output_path + class_name + "/" + class_name + "_x.npy"
        label_file = output_path + class_name + "/" + class_name + "_y.npy"
        np.save(feature_file, features)
        print("Saved " + feature_file)
        np.save(label_file, labels)
        print("Saved " + label_file)
    print("==========================================================DONE===================================================")
def preprocess_test_train(input_path = "", output_path = ""):
    bands = 60
    frames = 41
    if not os.path.exists( output_path ):
        os.mkdir( output_path, 0755 )
    class_names = get_class_names(path = input_path)
    print("Preprocessing...")
    for i , class_name in enumerate( class_names ):
        if not os.path.exists( output_path + class_name ):
            os.mkdir(output_path + class_name, 0755)
        if not os.path.exists( output_path + class_name + "/train/"):
            os.mkdir(output_path + class_name+ "/train/", 0755)
        if not os.path.exists( output_path + class_name + "/test/"):
            os.mkdir(output_path + class_name+ "/test/", 0755)
        for train_test in os.listdir(input_path + class_name):
            features, labels = extract_features( input_path, class_name , train_test, class_names, bands, frames )
            print ("Features of " + class_name + "_"+ train_test+":", features.shape)
            print ("Labels of " + class_name + "_"+ train_test+ ":", labels.shape)
            feature_file = output_path + class_name + "/" + train_test + "/" + class_name + "_x.npy"
            label_file = output_path + class_name + "/" + train_test + "/" + class_name + "_y.npy"
            np.save(feature_file, features)
            print("Saved " + feature_file)
            np.save(label_file, labels)
            print("Saved " + label_file)
    print("==========================================================DONE===================================================")
    
def main():
    # preprocess(input_path = "../../datasets/corpus/cnn_imeocap_all_dataset_agumented/", output_path = "../../datasets/features/cnn/cnn_augumented/")
    preprocess_test_train(input_path = "../../datasets/corpus/IEMOCAP_angry_sad_excited_neutral_test_train/", output_path = "../../datasets/features/cnn/cnn_test/")
main()

                






