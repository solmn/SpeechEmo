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
import librosa.display
from keras import backend as K
import scipy.io.wavfile as wav
class Test:
	def __init__(self, path, model, actual):
		import matplotlib.pyplot as plt
		self.plt = plt
		self.model = model
		self.path = path
		self.signal = []
		self.mfcc = []
		self.actual = actual
	
	def read_audio(self,filename):
		(sample_rate, signal) = wav.read(filename)
		y, sr = librosa.load(self.path, sr=16000)
		# print(signal.shape)
		
		# print(sr)
		self.signal = signal
		return signal, sample_rate
	def extract_mfcc(self,path):
	    samples, rate = self.read_audio(path)
	    # plt.plot(samples)
	    # plt.show()
	    features = mfcc(signal = samples, winlen=0.064, winstep=0.032, nfft=1024)
	    self.mfcc = features
	    x = []
	    for f in features:
	        x.append(f)
	    x = np.array(x)
	    features = []
	    features.append(x)
	    features = np.array(features)
	    features = sequence.pad_sequences(features, maxlen=200, padding='post', truncating='post')
	    return features
	def predict(self):
		font = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16}
		# global mf
		emotions = ["angry", "fear", "happy", "neutral", "sad","surprise"]
		# print(emotions)
		x = self.extract_mfcc(self.path)
		# print(self.path)
		x = (x - np.mean(x))/np.std(x)
		# mf = x
		# self.model.summary()
		inp = self.model.input                                           # input placeholder
		outputs = [layer.output for layer in self.model.layers]          # all layer outputs
		functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

		# Testing
		# test = np.random.random(input_shape)[np.newaxis,...]
		layer_outs = [func([x, 1.]) for func in functors]
		# for i in range (len(layer_outs)):
			# print(self.model.layers[i].name)
			# if(i !=-1):
			# 	data = layer_outs[i][0][0]
			# 	scaled = np.int16(data/np.max(np.abs(data)) * 32767)
			# 	wav.write("check" + str(i) + ".wav", 16000, scaled)
			# 	# librosa.display.specshow(layer_outs[i][0][0])
			# 	self.plt.plot(layer_outs[i][0][0])
			# 	self.plt.title(self.model.layers[i].name + str(layer_outs[i][0].shape))
			# 	self.plt.savefig(self.actual + " -  " + str(i) + ".png")
			# 	self.plt.show()
			# print(layer_outs[i][0].shape)
			# print("------------------------------------------------------------------------------------\n")
		prediction = self.model.predict(x)
		# print(prediction[0])
		max_index = 0;
		max_index_second = 0;
		max_val = prediction[0][0]
		max_val2 = 0;
		for i in range(len(prediction[0])):
			if(prediction[0][i] > max_val):
				max_val = prediction[0][i]
				max_index = i
			if(prediction[0][i] > max_val2):
				if(prediction[0][i] < max_val):
					max_val2 = prediction[0][i]
					max_index_second = i
		
		if(max_index != 3):
			print(emotions[max_index])
			# print("Second Candidate: " + emotions[max_index_second])
		else:
			# print(max_val, max_val2)
			if(max_val - max_val2 < 20):
				print(emotions[max_index_second])
				# print("Second Candidate: " + emotions[max_index])
			else:
				print(emotions[max_index])
				# print("Second Candidate: " + emotions[max_index_second])
		# print(max_val, max_val2)
		# fig1 = self.plt.figure(0)
		# plt1 = fig1.add_subplot(111)
		# plt1.plot(self.signal)
		# fig1.subplots_adjust(top=0.85)
		# plt1.text(1, 1,self.actual, fontdict=font)
		# # plt.show()
		# # print(self.mfcc)
		# fig2 = self.plt.figure(1)
		# plt2 = fig2.add_subplot(111)
		# plt2.plot(self.mfcc)
		# plt2.text(1, 1,self.actual, fontdict=font)
		# self.plt.show()
		# y, sr = librosa.load(self.path)
		# print(y.shape)
		# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
		# self.plt.figure(figsize=(10, 4))
		# # librosa.display.waveplot(y, sr=sr)
		
		# librosa.display.specshow(self.mfcc, x_axis='time')
		# # self.plt.imshow(self.mfcc.T, origin='lower', aspect='auto', interpolation='nearest')
		# self.plt.colorbar()
		# self.plt.title('MFCC')
		# self.plt.tight_layout()
		# self.plt.show()
		self.plot()
	def plot(self):
		y, sr = librosa.load(self.path)
		self.plt.figure()
		self.plt.subplot(1,1,1)
		librosa.display.waveplot(y, sr=sr)
		self.plt.tight_layout()
		

		self.plt.figure()
		self.plt.subplot(1, 1,1)
		wav.write("mfcc.wav", 16000, self.mfcc)
		librosa.display.specshow(self.mfcc, x_axis='time')
		# self.plt.show()

		

    # print(prediction[0])
def load_model():
	graph = "../../models/rnn/new/sequence_model_all_agumented.json"
	weight = "../../models/rnn/new/sequence_weight_all_agumented.h5"
	json_file = open(graph, 'r')
	loaded_model_json = json_file.read()
	json_file.close
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weight)
	return loaded_model
model = load_model()
# i = 0
# for layers in model.layers:
#     if(i==0):
#     		layer
path = "../../datasets/corpus/cnn_imeocap_all_dataset/happy/"
# import  os
f = os.listdir(path)
for i, fi in enumerate(f):
	test = Test(path + fi, model, "Happy")
	test.predict()
    # predict(path + fi, model)
#predict("h1.wav", model )
# predict("ang11.wav", model)
# predict("sa.wav", model)
# predict("fear1.wav", model)
# predict("neu.wav", model)
# predict("neu2.wav", model)
# test = Test("happy1.wav", model, "Happy")
# test.predict()

# test = Test("sa.wav", model, "S2")
# test.predict()

# test = Test("sad1.wav", model, "Sad")
# test.predict()

# plt.plot(mfcc)
# plt.show()
# predict("happy2.wav", model)
# predict("fear2.wav", model)
