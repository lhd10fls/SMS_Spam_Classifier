import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

nltk.download('stopwords')
nltk.download('punkt_tab')

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data/combined_spam_dataset.csv')

# load the combined dataset
combined_data = pd.read_csv(file_path)

# Preprocess the text data
STOPWORDS = stopwords.words('english') + ['u', 'im', 'c', 'y']
snow_stemmer = SnowballStemmer(language='english')

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def remove_stop_words(text: str) -> str:
    return ' '.join(word for word in text.split() if word not in STOPWORDS)

def clean_regex(text: str, pattern: str = "[^0-9a-zA-Z\s]+") -> str:
    return re.sub(pattern, '', text)

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stop_words(text)
    text = clean_regex(text)
    tokenized_text = word_tokenize(text, language='english')
    stem_words = [snow_stemmer.stem(word) for word in tokenized_text]
    return ' '.join(stem_words)

combined_data['text'] = combined_data['text'].apply(preprocess_text)

# transform text to vectors
# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(combined_data['text']).toarray()
y = combined_data['label'].values

# split data into train, test, validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)
print()

# train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluation
train_preds = knn_model.predict(X_train)
val_preds = knn_model.predict(X_val)
test_preds = knn_model.predict(X_test)

print("KNN TF-IDF 3000 Model Evaluation")
print("Train F1:", f1_score(y_train, train_preds))
print("Val F1:", f1_score(y_val, val_preds))
print("Test F1:", f1_score(y_test, test_preds))

print("Train bal acc:", balanced_accuracy_score(y_train, train_preds))
print("Val bal acc:", balanced_accuracy_score(y_val, val_preds))
print("Test bal acc:", balanced_accuracy_score(y_test, test_preds))

# save model
joblib.dump(knn_model, 'Model\Knn\knn_spam_classifier.pkl')
joblib.dump(tfidf, 'Model\Knn\\tfidf_knn_vectorizer.pkl')

# test the model
sample_text = [
    "Congratulations! You’ve won a $500 gift card. Click here to claim your prize.",
    "Let's meet tomorrow.",
    "Congratulations! You've won a free trip to Hawaii."
    ]
sample_processed = [preprocess_text(text) for text in sample_text]
sample_vector = tfidf.transform(sample_processed).toarray()
predictions = knn_model.predict(sample_vector)

for text, pred in zip(sample_text, predictions):
    print(f"Text: {text}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n") 