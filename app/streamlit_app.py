import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from app_logic import clean_text, load_glove_embeddings, analyze_sentiment, analyze_sentiment_with_lime  # updated import

def main():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter text below for a quick sentiment check. </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Typical computation time: 30 seconds to 3 minutes </p>", unsafe_allow_html=True)
    
    user_input = st.text_area("Enter text to analyze:")
    
    if st.button("Analyze"):
        st.write("Loading model and embeddings...")
        model = load_model("lstm_model.keras")
        glove_embeddings = load_glove_embeddings("glove.twitter.27B.200d.txt")
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        # Process sentiment first for quick display
        st.write("Cleaning text and predicting sentiment...")
        sentiment = analyze_sentiment(user_input, model, glove_embeddings, tfidf_vectorizer)
        
        # Conditional styling based on sentiment
        if sentiment == "Positive":
            st.markdown(f"<div style='background-color: #4CAF50; padding: 10px; color: white;'>Final Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        elif sentiment == "Negative":
            st.markdown(f"<div style='background-color: #f44336; padding: 10px; color: white;'>Final Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        else:  # Improved styling for neutral/other sentiment
            st.markdown(f"<div style='background-color: #FFC107; padding: 10px; border-radius: 5px; color: black;'>Final Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        
        # Now compute and display LIME scores
        st.info("Computing LIME scores...")
        # We can ignore the sentiment output from the following function call
        _, lime_scores = analyze_sentiment_with_lime(user_input, model, glove_embeddings, tfidf_vectorizer)
        st.write("LIME scores per embedded word:")
        st.write(lime_scores)

if __name__ == "__main__":
    main()
