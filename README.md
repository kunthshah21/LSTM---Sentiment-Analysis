# LSTM - Sentiment Analysis

A twitter sentiment analysis project built with LSTM.

## Files Overview

- **README.md**: Project documentation.
- **prediction.ipynb**: Contains the full workflow:
  - Data import and exploratory data analysis (EDA)
  - Cleaning and pre-processing text data
  - Converting text to embeddings using GloVe and TF-IDF
  - Label encoding for sentiment classification
  - Building, training, and evaluating an LSTM model for multi-class sentiment prediction
- **app/**: Contains the Streamlit application to serve the model.

## Model Approach

The model builds a robust sentiment analysis pipeline:
- **Data Pre-processing**: Removing null values, duplicates, and unwanted columns.
- **Text Cleaning**: Using regex and NLTK for removing URLs, mentions, non-word characters, and lemmatization.
- **Embedding Tweets**: Combining GloVe embeddings with TF-IDF weighting to create numerical representations of tweets.
- **Model Building**: A multi-layer LSTM model with BatchNormalization, Dropout, and regularization to predict sentiment.
- **Evaluation**: Classification report, confusion matrix, ROC-AUC per class, and MCC score are used for assessment.

## Running the Streamlit App

1. **Install Dependencies**:  
   Run the following command to install required packages.  
   ```
   pip install -r requirements.txt
   ```
2. **Run the App**:  
   Launch the Streamlit application by executing:  
   ```
   streamlit run app/app.py
   ``` 

3. **View the App**:  
   A browser window should open showing the interactive sentiment analysis interface.

## Requirements

Make sure you have Python 3.7+ installed. The project uses:
- numpy, pandas, matplotlib, seaborn
- nltk, scikit-learn
- TensorFlow / Keras
- streamlit
