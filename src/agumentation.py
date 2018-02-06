import soundfile as sf
import pyrubberband as pyrb
import os
import scipy.io.wavfile as wav
import random
class AudioAgumentation:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
    def do_agumentation(self):
        no_class = os.listdir(self.input_path)
        for name in no_class:
            files = os.listdir(self.input_path  + name + "/")
            files = [f for f in files if f.endswith(".wav")]
            for i, audio in enumerate(files):
                print(audio)
                y, sr = sf.read(self.input_path + name + "/" + audio)
                time = random.uniform(0.6, 1.3)
                y_strech = pyrb.time_stretch(y, sr, time)
                y_agument = pyrb.pitch_shift(y_strech, 22050, 1)
                # print(y_agument)
                wav.write(self.output_path + name + "/" +  "agumented_"+audio, sr, y_agument)
                print(name + "/"+"agumented_"+audio, "has augmented and saved")
a = AudioAgumentation("../datasets/corpus/cnn_imeocap_all_dataset/", "../datasets/corpus/cnn_imeocap_all_dataset_agumented/")
a.do_agumentation()
# rubberband  -t 1.5 -p 2.0 Happy_Ses01F_impro03_F000.wav Happy_Ses01F_impro03_F000_augmented.wav