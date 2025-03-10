# Movie-Genre-Prediction

# Overview
This project aims to predict the genre of a movie based on its plot summary. The model leverages multiple machine learning techniques, including LSTM (Long Short-Term Memory), CNN1D (1D Convolutional Neural Networks), and Naive Bayes to analyze and classify movie plot summaries into appropriate genres.

# Problem Statement
Given a movie plot summary, the goal is to predict the movie’s genre (e.g., Action, Comedy, Drama, etc.). The challenge lies in extracting relevant features from unstructured text data (the plot summaries) and applying machine learning techniques to make accurate genre predictions.

# Approach
## 1. Data Collection
The dataset used for this project is the 'wiki_movie_plots_deduped.csv' from Kaggle, that contains a collection of movies and their corresponding plot summaries. Each movie is labeled with its genre, which can be one or more categories (e.g., Comedy, Drama, etc.).

## 2. Preprocessing
Before training the models, the plot summaries undergo several preprocessing steps:

* Tokenization: Breaking down text into individual words.
* Stopword Removal: Eliminating common words like "and," "the," "in," etc.
* Padding: Ensuring all sequences have the same length for LSTM input.

## 3. Modeling
The project utilizes three different machine learning models for genre prediction:

### 3.1 LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) that is well-suited for sequential data like text. It is capable of remembering long-term dependencies in sequences, which is useful for understanding the context and sentiment of movie plot summaries.

### 3.2 CNN1D (1D Convolutional Neural Network)
CNNs are typically used in image processing, but they can also be applied to text data. A 1D CNN is used to learn spatial hierarchies in sequences, extracting important features from the plot summaries to make genre predictions.

### 3.3 Naive Bayes
A traditional machine learning algorithm based on Bayes' Theorem, Naive Bayes is used as a baseline model. It is simple yet effective for text classification tasks, where it computes the probability of each genre based on the frequency of words in the plot summaries.

## 4. Model Evaluation
The models are evaluated using accuracy, precision, recall, and F1-score. The best-performing model is selected based on these metrics.

# Dependencies
* Python 3.x
* TensorFlow (for LSTM and CNN1D)
* Scikit-learn (for Naive Bayes and other utilities)
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Matplotlib/Seaborn (for data visualization)
* NLTK (for text cleaning)
