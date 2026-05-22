import streamlit as st
import joblib
import sys

sys.path.append("src")

from preprocess import clean_text


# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# Page settings
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)


# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #FF9900;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown(
    '<div class="title">🛒 Amazon Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze Amazon product reviews using NLP and Machine Learning</div>',
    unsafe_allow_html=True
)


# Text input
review = st.text_area(
    "Enter your review below:",
    height=180,
    placeholder="Example: This product quality is amazing and worth the price..."
)


# Predict button
if st.button("Analyze Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a review first.")

    else:

        # Clean text
        cleaned_review = clean_text(review)

        # Vectorize
        review_vector = vectorizer.transform([cleaned_review])

        # Predict
        prediction = model.predict(review_vector)

        sentiment = (
            "Positive"
            if prediction[0] == 1
            else "Negative"
        )


        # Display result
        st.markdown("---")

        if sentiment == "Positive":

            st.success("😊 Positive Review")

            st.balloons()

        else:

            st.error("😞 Negative Review")