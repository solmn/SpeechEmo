from keras.models import model_from_json
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences





def load_model():
    graph = "../../models/rnn/sentiment_iemocap.json"
    weight = "../../models/rnn/sentiment_iemocap.h5"
    json_file = open(graph, 'r')
    loaded_model_json = json_file.read()
    json_file.close
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight)
    return loaded_model
sample = ["im sorry joy"]
X_train_encode = [one_hot(d, 3000) for d in sample]
X_train = pad_sequences(X_train_encode, maxlen=30, padding='post')
model = load_model()
prediction = model.predict(X_train)
print("Positive--------Neutral------------Negative")
print(prediction)
# for x in prediction[0]:
#     print(x)
# for x in prediction:
#     if(abs(x[0]*100-x[1]*100) < 1):
#         print("Neutral")
#     elif(x[0] > x[1]):
#         print("Positive")
#     elif(x[1]>x[0]):
#         print("Negative")
#     print(x)
