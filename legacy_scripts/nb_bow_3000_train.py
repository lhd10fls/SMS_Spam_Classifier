import pandas as pd
import os
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

nltk.download('stopwords')
nltk.download('punlt_tab')

# Path to the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data', 'combined_spam_dataset.csv')

# Load the dataset
combined_data = pd.read_csv(file_path)
print(combined_data.head())
print(combined_data.dtypes)

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

# BoW
bow = CountVectorizer(max_features=3000)  # Replaced TfidfVectorizer with CountVectorizer
X = bow.fit_transform(combined_data['text']).toarray()  # Convert to dense array
y = combined_data['label'].values


# Split dataset (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

# Naive Bayes with GridSearchCV
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}

nb_model = MultinomialNB()
grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score (f1):", grid_search.best_score_)

# Retrain best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluation
train_preds = best_model.predict(X_train)
val_preds = best_model.predict(X_val)
test_preds = best_model.predict(X_test)

print("Naive Bayes BoW 3000 Model Evaluation")
print("Train F1:", f1_score(y_train, train_preds))
print("Val F1:", f1_score(y_val, val_preds))
print("Test F1:", f1_score(y_test, test_preds))

print("Train bal acc:", balanced_accuracy_score(y_train, train_preds))
print("Val bal acc:", balanced_accuracy_score(y_val, val_preds))
print("Test bal acc:", balanced_accuracy_score(y_test, test_preds))

print("Classification Report (Test):\n", classification_report(y_test, test_preds))

# Save model and vectorizer
nb_model_path = os.path.join(script_dir, 'Model', 'Nb', 'naive_bayes_spam_model.pkl')
vec_path = os.path.join(script_dir, 'Model', 'Nb', 'bow_vectorizer_1000.pkl')

os.makedirs(os.path.dirname(nb_model_path), exist_ok=True)
joblib.dump(best_model, nb_model_path)
joblib.dump(bow, vec_path)

# Test the model on sample input
sample_text = ["Win a free iPhone now!", "Let's meet tomorrow."]
sample_processed = [preprocess_text(text) for text in sample_text]
sample_vector = bow.transform(sample_processed).toarray()
predictions = best_model.predict(sample_vector)

for text, pred in zip(sample_text, predictions):
    print(f"Text: {text}\nPrediction: {'Spam' if pred == 1 else 'Ham'}\n")
