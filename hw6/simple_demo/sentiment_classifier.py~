__author__ = 'xead'
import pickle

class SentimentClassifier(object):
    def __init__(self):
        with open('model', 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open('vectorizer', 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        self.classes_dict = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", -1: "prediction error"}

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(vectorized)[0]
            return prediction
        except:
            print("prediction error")
            return -1

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized)
        except:
            print('prediction error')
            return -1

    def get_prediction_message(self, text):
        class_prediction = self.predict_text(text)
        return "Predicted rating: " + self.classes_dict[class_prediction]
