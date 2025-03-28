import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer  # new import

#########################################
# Text Cleaning Function
#########################################
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()  # convert to lowercase
    text = re.sub(r'@\w+', '', text)  # remove any @mentions
    text = re.sub(r'#', '', text)  # remove # from hashtags
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ascii characters
    text = re.sub(r'\d', '', text)  # remove digits
    text = re.sub(r'\s+', ' ', text)  # fix multiple spacing
    text = re.sub(r'^\s+|\s+?$', '', text)  # trim leading/trailing spaces

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    return text

#########################################
# GloVe Embeddings Loader
#########################################
def load_glove_embeddings(filepath):
    glove_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_dict[word] = vector
    return glove_dict


#########################################
# Compute TF-IDF Weighted GloVe Embedding
#########################################
def get_tweet_embedding(tweet, glove_dict, tfidf_scores, feature_names):
    words = tweet.split()
    tweet_vector = np.zeros(200)  # using GloVe 200d embeddings
    weight_sum = 0.0

    for word in words:
        if word in glove_dict and word in feature_names:
            # Use the TF-IDF weight from the fitted vectorizer; default to 1 if not found
            weight = tfidf_scores.get(word, 1.0)
            tweet_vector += weight * glove_dict[word]
            weight_sum += weight

    return tweet_vector / weight_sum if weight_sum != 0 else tweet_vector


#########################################
# Main function to run sentiment analysis
#########################################
def main():
    # Load the trained sentiment analysis model
    model = load_model("lstm_model.keras")

    # Load pre-trained GloVe embeddings (200d)
    glove_path = "glove.twitter.27B.200d.txt"
    glove_embeddings = load_glove_embeddings(glove_path)

    # Load the fitted TF-IDF vectorizer from file
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Extract TF-IDF idf scores and feature names
    idf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    feature_names = set(tfidf_vectorizer.get_feature_names_out())

    # Define a random sample text
    sample_text = "@mystic you are a terrible person"
    
    # Process the sample text
    print("Original Text: ", sample_text)
    cleaned_text = clean_text(sample_text)
    print("Cleaned Text: ", cleaned_text)

    # Generate embedding for the cleaned text
    tweet_embedding = get_tweet_embedding(cleaned_text, glove_embeddings, idf_scores, feature_names)
    # Reshape embedding to match model input (expected shape: (1, 1, 200))
    tweet_embedding = tweet_embedding.reshape(1, 1, 200)

    # Predict sentiment using the loaded model
    prediction = model.predict(tweet_embedding)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Define mapping for classes: 0: Negative, 1: Neutral, 2: Positive
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = mapping.get(predicted_class, "Unknown")

    # Print the prediction result
    print("Predicted Sentiment: ", sentiment)

def analyze_sentiment(text, model, glove_embeddings, tfidf_vectorizer):
    cleaned_text = clean_text(text)
    idf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    feature_names = set(tfidf_vectorizer.get_feature_names_out())
    tweet_embedding = get_tweet_embedding(cleaned_text, glove_embeddings, idf_scores, feature_names)
    tweet_embedding = tweet_embedding.reshape(1, 1, 200)
    prediction = model.predict(tweet_embedding)
    predicted_class = np.argmax(prediction, axis=1)[0]
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping.get(predicted_class, "Unknown")

def analyze_sentiment_with_lime(text, model, glove_embeddings, tfidf_vectorizer):
    # Build a prediction function for LIME which expects a list of texts.
    def predict_proba(texts):
        results = []
        for t in texts:
            cleaned = clean_text(t)
            idf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
            feature_names = set(tfidf_vectorizer.get_feature_names_out())
            embedding = get_tweet_embedding(cleaned, glove_embeddings, idf_scores, feature_names)
            embedding = embedding.reshape(1, 1, 200)
            pred = model.predict(embedding)[0]
            results.append(pred)
        return np.array(results)
    
    explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
    explanation = explainer.explain_instance(text, predict_proba, num_features=len(text.split()))
    probs = predict_proba([text])[0]
    predicted_class = np.argmax(probs)
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = mapping.get(predicted_class, "Unknown")
    lime_scores = explanation.as_list(label=predicted_class)
    return sentiment, lime_scores

if __name__ == "__main__":
    main()
