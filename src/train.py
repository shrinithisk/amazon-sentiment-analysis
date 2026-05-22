import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocess import clean_text


# Load processed dataset
df = pd.read_csv("data/processed.csv")


# Keep required columns
df = df[['Text', 'Score', 'Cleaned_Text']]


# Remove null values
df.dropna(inplace=True)


# Create sentiment labels
df['Sentiment'] = df['Score'].apply(
    lambda x: 0 if x <= 2 else 1
)


# Features and labels
X = df['Cleaned_Text']

y = df['Sentiment']


# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LogisticRegression()

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Save model
joblib.dump(model, "models/model.pkl")


# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")


print("Model and vectorizer saved successfully!")

