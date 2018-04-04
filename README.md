# SpeechEmo
- Emotional context classification from acoustic feature of speech signal

- Contains Different version of deep learning networks and trained models written in Keras backend tensorflow.
  - CNN 
  - LSTM
  - CNN_LSTM
  - CTC with BLSTM

**Dependancies**

* Tensorflow >=1
* keras >= 2.0
* numpy
* librosa 0.4
* scipy
* [python_speech_features](https://github.com/jameslyons/python_speech_features)


**Datasets** 

I have used Interactive Emotional Dyadic Motion Capture ([IEMOCAP](http://sail.usc.edu/iemocap/)) database. It is is an acted, multimodal and multispeaker database

**How to label/format the dataset**

IEMOCAP database is recorded in different session,each session contains a set of audios and corresponding evalutation script that helps us to segment the dataset based on emotional categories.

- Change the directory to  ```src\utils``` folder and specify your IEMOCAP dataset path and output path in the utils.py 

```
   cd src/utils/
   python utils.py
```
- To split the dataset to train and test set

```
   python split.py 
```

**Feature Extraction and Training**
- Extract acoustic features(spectrogram and MFCC features) and train the models.

```
    # CNN model
    cd src/cnn/
    python cnn_feat_extract.py
    python cnn_train.py
    
    # LSTM model
    cd src/rnn/
    python rnn_feat.py
    python cnn_lstm_modle.py
    
    
    # CNN_LSTM
    cd src/cnn_lst
    python feature_extract.py
    python rnn_tray.py
    
    #CTC modle
    cd src/ctc/
    python utils/util.py
    python ctc_tray.py
```



