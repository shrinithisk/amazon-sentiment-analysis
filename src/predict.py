import joblib

from preprocess import clean_text


# Load saved model
model = joblib.load("../models/model.pkl")


# Load vectorizer
vectorizer = joblib.load("../models/vectorizer.pkl")


def predict_sentiment(review):

    review = clean_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        return "Positive"

    else:
        return "Negative"


# Test prediction
sample_review = "This product is really good"

result = predict_sentiment(sample_review)

print(result)