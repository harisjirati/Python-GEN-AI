import pickle


model = pickle.load(open("notebook/model.pkl", "rb"))
vectorizer = pickle.load(open("notebook/vectorizer.pkl", "rb"))

def predict_spam(text):
    text_clean = text.lower()
    text_vec = vectorizer.transform([text_clean])
    return model.predict(text_vec)[0]

msg = input("Enter message: ")
print("Prediction:", predict_spam(msg))