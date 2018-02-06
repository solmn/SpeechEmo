import os
import numpy as np
import re
class util:
    def __init__(self):
        self.path = "../../datasets/corpus/IEMOCAP/sessions/"
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.emotions = ['exc', 'neu','sad','ang', 'hap', 'sur','fea']
        self.labels = {'exc':"Positive", 'hap':"Positive", 'sur':"Positive", 'neu':"Neutral", 'sad':"Negative", 'fru':"Negative", 'ang':"Negative", 'fea':"Negative", 'dis':"Negative"}
    def read_iemocap_data(self):
        X = []
        Y = []
        all_transcription = []
        all_emotions = []
        for session in self.sessions:
            path_to_transcriptions = self.path + session + '/dialog/transcriptions/'
            path_to_emotions = self.path + session + '/dialog/EmoEvaluation/'
            files = os.listdir(path_to_emotions)
            files = [f for f in  files if f.endswith('.txt')]
            for f in files:
                transcriptions = self.get_transcription(path_to_transcriptions, f)
                all_transcription += transcriptions
                emotions = self.get_emotion(path_to_emotions, f)
                all_emotions += emotions
        for ts in all_transcription:
            for emo in all_emotions:
                if(ts['id'] == emo['id']):
                    if(emo['emotion'] in self.emotions):
                       X.append(self.cleanSentences(ts['ts']))
                       Y.append(self.labels[emo['emotion']])
        
            
        return X, Y
        # p = 0
        # n = 0
        # ne = 0
        # for l in Y:
        #     if(l == "Positive"):
        #         p+=1
        #     elif(l =="Negative"):
        #         n +=1
        #     elif(l == "Neutral"):
        #         ne +=1
        # print("Positive = " + str(p))
        # print("Negative = " + str(n))
        # print("Neutral " + str(ne))
       
                    
               
    def cleanSentences(self,string):
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower()) 
    def get_transcription(self,path, filename):
        f = open(path + filename, 'r').read()
        f = np.array(f.split('\n'))
        transcription_info = []
        for i in range(len(f) - 1):
            transcription = {}
            line = f[i]
            i1 = line.find(': ')
            i0 = line.find(' [')
            t0 = i0
            t1 = line.find(']: ')
            id = line[:i0]
            ts = line[i1 + 2 : ]
            time = line[t0 + 2 : t1]
            transcription['id'] = id
            transcription['ts'] = ts.strip()
            transcription['time'] = time
            transcription_info.append(transcription)
        return transcription_info
            
        
    def get_emotion(self,path, filename):
        f = open(path + filename, 'r').read()
        f = np.array(f.split('\n'))
        idx = f == ''
        idx_n = np.arange(len(f))[idx]
        emotions = []
        for i in range(len(idx_n) - 2):
            emo = {}
            emo_line = f[idx_n[i] + 1]
            t0 = emo_line.find('[')
            t1 = emo_line.find(']')
            emo['time'] = emo_line[t0 +1 :t1]
            i0 = emo_line.find(filename[:-4])
            i1 = emo_line.find(filename[:-4]) + len(filename[:-4]) + 5
            emo['id'] = emo_line[i0:i1]
            e0 = emo_line.find('\t[')
            emo['emotion'] = emo_line[e0 - 3:e0]
            emotions.append(emo)
        return emotions
                        

