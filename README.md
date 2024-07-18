Sentiment Analysis of IMDB Movie Reviews

# IMDB Sentiment Analysis with RNN

This project uses a simple Recurrent Neural Network (RNN) to predict the sentiment of movie reviews from the IMDB database.

## Project Overview

The goal of this project is to build a sentiment analysis model that can classify the sentiment of IMDB movie reviews as positive or negative. We use an RNN to process the text data and make predictions.

## Dataset

The dataset used in this project is the IMDB movie reviews dataset. It contains 50,000 reviews, with 25,000 for training and 25,000 for testing. Each review is labeled as either positive or negative.

## Model Architecture

We use a simple RNN architecture for this project. The model consists of the following layers:
- Embedding Layer: Converts words to dense vectors of fixed size.
- RNN Layer: Processes the sequence of word vectors.
- Dense Layer: Produces the final output.

## Installation

To run this project, you'll need Python and the following libraries:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib
