# Email Spam Detection using Bayesian Machine Learning

## Abstract
Email spam detection is a classic problem in Natural Language Processing (NLP) and probabilistic machine learning. The objective is to automatically classify incoming messages as Spam (unsolicited or malicious emails) or Ham (legitimate emails).

This project implements a Naive Bayes based probabilistic classifier combined with TF-IDF feature extraction to detect spam messages. The system includes a full machine learning pipeline consisting of data preprocessing, feature engineering, model training, and real-time prediction through a Streamlit web interface.

The work demonstrates the effectiveness of Bayesian learning methods for text classification problems and provides a reproducible implementation suitable for experimentation and research exploration.

---

## Problem Statement

Given an email message `x`, classify it into one of the following categories:

y ∈ {Spam, Ham}

The goal is to estimate the probability:

P(y | x)

and assign the class with the highest posterior probability.

---

## Methodology

The system follows a probabilistic machine learning pipeline.

Pipeline Overview

Raw Email Text  
      │  
      ▼  
Text Preprocessing  
(cleaning, tokenization)  
      │  
      ▼  
Feature Extraction  
(TF-IDF Vectorization)  
      │  
      ▼  
Naive Bayes Classifier  
(Multinomial NB)  
      │  
      ▼  
Spam / Ham Prediction

---

## Mathematical Formulation

The model is based on Bayes' Theorem:

P(y|x) = (P(x|y) * P(y)) / P(x)

Where:

P(y|x) = Posterior probability of class y  
P(x|y) = Likelihood of observing text x given class y  
P(y) = Prior probability of class  
P(x) = Evidence  

Under the Naive Independence Assumption, the likelihood becomes:

P(x|y) = ∏ P(xᵢ | y)

where xᵢ represents individual words in the message.

The classifier predicts:

ŷ = argmax P(y) ∏ P(xᵢ|y)

For numerical stability, log probabilities are used:

log P(y|x) = log P(y) + Σ log P(xᵢ|y)

---

## Feature Engineering

Text data is transformed into numerical representations using TF-IDF (Term Frequency – Inverse Document Frequency).

### TF (Term Frequency)

TF(t,d) = (Number of occurrences of term t) / (Total terms in document)

### IDF (Inverse Document Frequency)

IDF(t) = log (N / (1 + df(t)))

Where

N = number of documents  
df(t) = number of documents containing term t  

Final feature weight:

TFIDF(t,d) = TF(t,d) × IDF(t)

This representation emphasizes informative words while reducing the weight of common words.

---

## System Architecture

Email Dataset  
      │  
      ▼  
Text Preprocessing  
(cleaning, lowercasing, stopword removal)  
      │  
      ▼  
TF-IDF Vector Representation  
      │  
      ▼  
Naive Bayes Classifier  
      │  
      ▼  
Spam / Ham Output  
      │  
      ▼  
Streamlit Web Interface

---

### File Description

- **spam.csv**  
  Contains labeled email or SMS messages used for training the machine learning model.

- **spam_model.pkl**  
  Stores the trained Naive Bayes model serialized using pickle.

- **train
- .py**  
  Script responsible for training the machine learning model using the dataset and saving the trained model.

- **app.py**  
  Streamlit web application that allows users to input email text and receive spam classification results.

- **requirements.txt**  
  Contains all Python dependencies required to run the project.

---

## Dataset

The dataset contains labeled messages used for supervised learning.

Example records:

Label: ham  
Message: Are we still meeting today?

Label: spam  
Message: Congratulations! You have won a free prize!

Classes:

- **Ham (0)** → Legitimate messages  
- **Spam (1)** → Unwanted or promotional messages

Dataset Characteristics:

- Binary classification dataset
- Text-based messages
- Used for Natural Language Processing tasks

---

## Implementation Details

### Text Preprocessing

The preprocessing stage includes:

- Lowercasing text
- Removing punctuation
- Removing stopwords
- Tokenization
- Feature vectorization

These steps ensure the text is transformed into a clean and usable format for machine learning models.

---

### Model Training

The classifier used is **Multinomial Naive Bayes**, which works well for text classification problems using word frequencies or TF-IDF features.

Training pipeline:

Dataset → Preprocessing → TF-IDF Vectorization → Naive Bayes Classifier → Model Serialization

The trained model is saved using **pickle** so it can be reused without retraining.

---

## Installation

Clone the repository : https://github.com/Rushithadatta/email_spam_detection

Navigate to the project directory:


cd email_spam_detection


Install dependencies:


pip install -r requirements.txt


---

## Train the Model

Run the training script:


python train_model.py


This will:

- Load the dataset
- Preprocess the text data
- Train the Naive Bayes classifier
- Save the trained model in the `model/` folder

---

## Run the Application

Start the Streamlit application:


streamlit run app.py


The application will open in your browser where you can enter an email message and check if it is **Spam** or **Ham**.

---

## Example Prediction

Input message:


Congratulations! You have won a free iPhone. Click here to claim.


Output:


Spam


---

## Evaluation Metrics

Typical metrics used to evaluate spam classification models include:

- Accuracy
- Precision
- Recall
- F1 Score

Precision:


Precision = TP / (TP + FP)


Recall:


Recall = TP / (TP + FN)


F1 Score:


F1 = 2 * (Precision * Recall) / (Precision + Recall)


These metrics are important for evaluating classification performance, especially in imbalanced datasets.

---

## Applications

Spam detection systems are widely used in:

- Email filtering systems
- SMS spam detection
- Phishing detection
- Social media content moderation
- Cybersecurity threat detection

---

## Future Improvements

Possible enhancements for this project include:

- Using deep learning models such as LSTM or BERT
- Increasing dataset size for better generalization
- Deploying the system using Docker and cloud platforms
- Implementing multilingual spam detection
- Integrating with real-time email servers

---

## References

- Christopher M. Bishop — *Pattern Recognition and Machine Learning*  
- Jurafsky & Martin — *Speech and Language Processing*  
- Scikit-learn Documentation  
- Manning, Raghavan & Schütze — *Introduction to Information Retrieval*

---

## Author

**Rushitha Datta**  
Computer Science Student  

Focus Areas:

- Machine Learning
- Data Structures and Algorithms
- Probabilistic Models
- Full Stack Development

