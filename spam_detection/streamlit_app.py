import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("notebook/model.pkl", "rb"))
vectorizer = pickle.load(open("notebook/vectorizer.pkl", "rb"))

# Title
st.title("📩 Spam Message Classifier")

# Description
st.write("Enter a message to check if it is Spam or Ham")

# Input box
user_input = st.text_area("Enter your message:")

# Button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        # Preprocess
        text = user_input.lower()
        text_vec = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(text_vec)[0]

        # Output
        if prediction == "spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM (HAM)")