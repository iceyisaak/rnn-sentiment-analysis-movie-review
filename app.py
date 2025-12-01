import numpy as np
import streamlit as st
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model



ds = imdb

word_index = ds.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple-rnn_movie-review-analysis.keras')


##################################################################################

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = pad_sequences([encoded_review], max_len=500)
    return padded_review


def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    pred = model.predict(preprocess_input)

    prediction = pred[0][0]
    sentiment_vibe  = 'Positive' if prediction > 0.5 else 'Negative'

    return sentiment_vibe, prediction


##################################################################################


st.title('Sentiment Analysis: IMDB Movie Review')
st.write('Enter a movie review to classify it as Positive or Negative')

text_input = st.text_area('Movie Review')