# IMDb Movie Review Sentiment Analysis using Simple RNN

An NLP-based sentiment analysis project that uses a Simple Recurrent Neural Network (RNN) to classify IMDb movie reviews as positive or negative.

## Overview

This project uses the IMDb movie review dataset to train a Simple RNN model for binary sentiment classification.

The model processes text reviews, learns patterns from the sequence of words, and predicts whether a review expresses a positive or negative sentiment.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- NLP
- Simple RNN
- IMDb Dataset

## Workflow

1. Load the IMDb dataset
2. Preprocess the movie reviews
3. Tokenize the text
4. Convert words into numerical sequences
5. Pad sequences to a fixed length
6. Build a Simple RNN model
7. Train the model
8. Evaluate the model
9. Save the trained model
10. Use the model to predict sentiment for new reviews

## Model

The project uses a Simple Recurrent Neural Network to process the sequential nature of text.

The general architecture includes:

- Embedding layer
- Simple RNN layer
- Dense output layer
- Sigmoid activation for binary classification

## Prediction

The trained model accepts a movie review as input and produces a prediction score.

Example:

**Input:**

> This movie was amazing. The story was excellent and I really enjoyed it.

**Output:**

```text
Sentiment: Positive
