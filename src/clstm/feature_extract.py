import librosa
import os
import numpy as np
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Neutral']
input_path = "../../datasets/corpus/IEMOCAP_angry_sad_excited_neutral_test_train/"
output_path = "../../datasets/features/cnn_lstm_features/"
def get_class_names ( path=""):
    class_names = os.listdir(path)
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

def load_audio(path):
    signal, samplerate = librosa.load(path, mono=True)
    return signal

def extract_spectogram_feat(signal, class_name, class_names):
    frames = 41
    bands = 60
    window_size = 512 * (frames -1)
    sequence = []
    label = []
    for ( start, end ) in windows( signal, window_size ):
            if ( len(signal[start:end]) == int(window_size) ):
                audio_signal = signal[start:end]
                mel_spec = librosa.feature.melspectrogram(audio_signal, n_mels = bands)
                log_spec = librosa.logamplitude(mel_spec)
                delta_log_spec = librosa.feature.delta(log_spec)
                seq = []
                seq = log_spec.reshape(log_spec.shape[0], log_spec.shape[1], 1)
                seq = np.concatenate((seq, np.zeros(np.shape(seq))), axis=2)
                seq[:,:,1] = delta_log_spec
                sequence.append(seq)
    sequence = np.asarray(sequence)
    label.append(encode_class(class_name, class_names))
   
    print(sequence.shape)
    return sequence, label
def pre_process(input = "", output = ""):
    create_directory(output_path)
    for i, class_name in enumerate(EMOTIONS):
        create_directory(output_path + class_name)
        create_directory(output_path + class_name)
        train_test = os.listdir(input_path + class_name)
        for folder in train_test:
            print("Processing: " + class_name + ": " + folder  + "..." )
            sequence = []
            labels = []
            create_directory(output_path + class_name + "/" + folder + "/")
            path = input_path + class_name + "/" + folder + "/"
            wav_files = os.listdir(path)
            wav_files = [f for f in wav_files if f.endswith(".wav")]
            feature_path = output_path + class_name + "/" + folder + "/"
            for i, file in enumerate(wav_files):
                file_path = path + file
                audio = load_audio(file_path)
                features, label = extract_spectogram_feat(audio, class_name, EMOTIONS)
                if(features.shape[0] != 0):
                    sequence.append(features)
                    labels.append(label)
                else:
                    print("Got you: ", features.shape)
            sequence = np.asarray(sequence)
            print("Hooly shit", sequence.shape)
            labels = np.asarray(labels)
            save_feature(feature_path, sequence, labels)
            print(class_name + ":" + folder + " .... Saved features.")
def save_feature(path, sequence, labels):
    np.save(path + "_X.npy", sequence)
    np.save(path + "_Y.npy", labels)

            
def create_directory(path):
    if not os.path.exists(path):
        os.mkdir( path, 0755 )
pre_process(input = input_path, output = output_path)

