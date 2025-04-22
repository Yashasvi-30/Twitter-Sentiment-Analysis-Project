import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
import re

# Download necessary NLTK resources
nltk.download('stopwords')




# Load the saved model and tokenizer
model = tf.keras.models.load_model("lstm_sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit App
def predict_sentiment(text):
    # Clean the text input
    cleaned_text = clean_text(text)
    
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map prediction to sentiment label
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_labels[predicted_class]

# Streamlit UI
st.title("Twitter Sentiment Analysis")

st.markdown("""
    This app uses a pre-trained LSTM model to predict the sentiment of a given tweet.
    The sentiment can be Negative, Neutral, or Positive.
""")

# User input text
user_input = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"The predicted sentiment is: {sentiment}")
    else:
        st.warning("Please enter a tweet to predict the sentiment.")

# Footer Section
st.markdown("""
    <div class="footer">
        <p>Created with ❤️ by Yashasvi Gupta | <a href="https://github.com/Yashasvi-30/Twitter-Sentiment-Analysis-Project" target="_blank">GitHub</a></p>
    </div>
""", unsafe_allow_html=True)
