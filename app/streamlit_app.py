import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from app_logic import clean_text, load_glove_embeddings, analyze_sentiment

def main():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter text below for a quick sentiment check.</p>", unsafe_allow_html=True)
    
    user_input = st.text_area("Enter text to analyze:")
    
    
    if st.button("Analyze"):
        st.write("Loading model and embeddings...")
        model = load_model("lstm_model.keras")
        glove_embeddings = load_glove_embeddings("glove.twitter.27B.200d.txt")
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        st.write("Cleaning text...")
        st.write("Applying embeddings...")
        st.write("Predicting sentiment...")

        sentiment = analyze_sentiment(user_input, model, glove_embeddings, tfidf_vectorizer)
        st.success(f"Final Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
