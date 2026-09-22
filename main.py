import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#step 1:Load the imdb dataset
word_index = imdb.get_word_index()
reversed_word_index = {value : key for key, value in word_index.items()}

#Load the pre-trained model with RELU Activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = []

    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)  # unknown word

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review

import streamlit as st
##Streamlit App

st.title('IMDB Movie Review Analysis')
st.write('Enter a movie Review to classify it as positive or negative')

#User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    #Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write('Please Enter a Movie review')