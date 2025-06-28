import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI title
st.title("🎬 Movie Review Sentiment Analyzer")

# User input
review = st.text_area("Enter a movie review", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]
        sentiment = "positive" if pred == 1 else "negative"
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** {prob:.2f}")