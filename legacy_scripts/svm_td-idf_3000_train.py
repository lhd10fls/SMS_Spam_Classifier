import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Preprocessing data

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

# train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Evaluation
train_preds = svm_model.predict(X_train)
val_preds = svm_model.predict(X_val)
test_preds = svm_model.predict(X_test)

print("SVM TF-IDF 3000 Model Evaluation")
print("Train F1:", f1_score(y_train, train_preds))
print("Val F1:", f1_score(y_val, val_preds))
print("Test F1:", f1_score(y_test, test_preds))

print("Train bal acc:", balanced_accuracy_score(y_train, train_preds))
print("Val bal acc:", balanced_accuracy_score(y_val, val_preds))
print("Test bal acc:", balanced_accuracy_score(y_test, test_preds))

print("Classification Report (Test):\n", classification_report(y_test, test_preds))

# save model
joblib.dump(svm_model, 'Model\Svm\svm_spam_classifier.pkl')
joblib.dump(tfidf, 'Model\Svm\\tfidf_svm_vectorizer.pkl')

# test the model
sample_text = ["Win a free iPhone now!", "Let's meet tomorrow."]
sample_processed = [preprocess_text(text) for text in sample_text]
sample_vector = tfidf.transform(sample_processed).toarray()
predictions = svm_model.predict(sample_vector)

for text, pred in zip(sample_text, predictions):
    print(f"Text: {text}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")