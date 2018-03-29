from keras.models import model_from_json
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

def load_model():
    # graph = "../../models/rnn/sentiment_iemocap.json"
    # weight = "../../models/rnn/sentiment_iemocap.h5"
    graph = "../../models/rnn/sentiment2.json"
    weight = "../../models/rnn/sentiment2.h5"
    json_file = open(graph, 'r')
    loaded_model_json = json_file.read()
    json_file.close
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight)
    return loaded_model
sample = ["that is bad"]
X_train_encode = [one_hot(d, 2000) for d in sample]
X_train = pad_sequences(X_train_encode, maxlen=200, padding='post')
model = load_model()
prediction = model.predict(X_train)
# print("Positive-------Negative")
# print(prediction)
# for x in prediction[0]:
#     print(x)
print(prediction)
print("predicting emotion from text")

print("\n\n")
for x in prediction:
    if(abs(x[0]*100-x[1]*100) < 1):
        print("Neutral")
    if(x[0] > x[1]):
        print("Emotion: Positive")
    elif(x[1]>x[0]):
        print("Emotion: Negative")
    print(x)
print("\n")
