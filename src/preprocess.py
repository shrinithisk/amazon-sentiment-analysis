import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords + Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)


# Load dataset
df = pd.read_csv("data/Reviews.csv")

# Apply preprocessing
# Change column name if needed
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Save processed dataset
df.to_csv("data/processed.csv", index=False)

print("Processed data saved successfully!") 