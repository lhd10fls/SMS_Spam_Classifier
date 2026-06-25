import tkinter as tk
from tkinter import ttk
import os
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle
with open("Model/Lstm//vocab.pkl", "rb") as f:
    vocab = pickle.load(f)


# === Download NLTK resources (only once) ===
nltk.download('stopwords')
nltk.download('punkt')

# === Preprocessing setup ===
STOPWORDS = stopwords.words('english') + ['u', 'im', 'c', 'y']
snow_stemmer = SnowballStemmer(language='english')
stop_words = set(ENGLISH_STOP_WORDS)

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers = 1, dropout_rate = 0.3):
        super(LSTM_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_dim
        )

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers = self.num_layers,
                          dropout = self.dropout_rate, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # X has shape (batch_size, max_len)
        # embedding has shape (batch_size, max_len, embedding_dim)
        embedding = self.dropout(self.embedding(x))

        # LSTM has output of shape (batch_size, max_len, hidden_dim)
        # Hidden has output of shape (batch_size, num_layers, hidden_dim)
        # Hidden is the hidden state of the last cell of every layers
        lstm_output, hidden = self.lstm(embedding)

        # Taking only the output of the last LSTM cell
        # output_logits has shape (batch_size, 1, hidden_dim)
        # After squeezing, this should have shape (batch_size, hidden_dim)
        output_logits = self.fc(lstm_output[:, -1, :])

        # Outputs should have shape (batch_size, 2)
        return output_logits.squeeze()
    
vocab_size = len(vocab)
embedding_dim = 200
hidden_dim = 200
num_layers = 1



UNK_IDX, PAD_IDX = 0, 1
SPECIAL_SYMBOLS = ['<unk>', '<pad>']

def text_to_tensor(text, vocab, tokenizer, device, pad_idx=1):
    tokens = tokenizer(text)
    if len(tokens) == 0:
        tokens = ['<pad>'] * 10
    ids = vocab.lookup_indices(tokens)
    if len(ids) == 0:
        ids = [pad_idx] * 10
    tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    return tensor


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

# === Load model and vectorizer ===
script_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(script_dir, 'Model', 'Nb', 'naive_bayes_spam_model.pkl'))
vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Nb', 'tfidf_vectorizer_1000.pkl'))

# === GUI logic ===
def process_input():
    input_text = input_box.get("1.0", tk.END).strip()
    selected_model = model_selector.get()

    if selected_model == "LR":
        model = joblib.load(os.path.join(script_dir, 'Model', 'Lr', 'logreg_spam_model.pkl'))
        vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Lr', 'tfidf_vectorizer_1000.pkl'))
        processed = preprocess_text(input_text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        output_text = "Spam" if prediction == 1 else "Ham"
    elif selected_model == "KNN":
        model = joblib.load(os.path.join(script_dir, 'Model', 'Knn', 'knn_spam_classifier.pkl'))
        vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Knn', 'tfidf_knn_vectorizer.pkl'))
        processed = preprocess_text(input_text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        output_text = "Spam" if prediction == 1 else "Ham"
    elif selected_model == "SVM":
        model = joblib.load(os.path.join(script_dir, 'Model', 'Svm', 'svm_spam_classifier.pkl'))
        vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Svm', 'tfidf_svm_vectorizer.pkl'))
        processed = preprocess_text(input_text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        output_text = "Spam" if prediction == 1 else "Ham"
    elif selected_model == "NB":
        model = joblib.load(os.path.join(script_dir, 'Model', 'Nb', 'naive_bayes_spam_model.pkl'))
        vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Nb', 'tfidf_vectorizer_1000.pkl'))
        processed = preprocess_text(input_text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        output_text = "Spam" if prediction == 1 else "Ham"
    elif selected_model == "LR":
        model = joblib.load(os.path.join(script_dir, 'Model', 'Lr', 'logistic_regression_spam_model.pkl'))
        vectorizer = joblib.load(os.path.join(script_dir, 'Model', 'Lr', 'tfidf_vectorizer_1000.pkl'))
        processed = preprocess_text(input_text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        output_text = "Spam" if prediction == 1 else "Ham"
    elif selected_model == "LSTM":
        model = LSTM_model(vocab_size, embedding_dim, hidden_dim, num_layers)
        model.load_state_dict(torch.load("Model\Lstm\LSTM_model (1).pth", map_location=device))
        model.to(device)
        model.eval()
        text = input_text  
        tensor = text_to_tensor(text, vocab, tokenizer, device)
        output = model(tensor)
        prediction = torch.sigmoid(output).item()
        output_text = "Spam" if prediction > 0.5 else "Ham"


    else:
        output_text = "Invalid model selected."

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, output_text)
    output_box.config(state=tk.DISABLED)

# === GUI Layout ===
root = tk.Tk()
root.title("Text Processor with Model Selection")
root.geometry("500x450")

ttk.Label(root, text="Input:").pack(anchor=tk.W, padx=10, pady=(10, 0))
input_box = tk.Text(root, height=6, wrap=tk.WORD)
input_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

ttk.Label(root, text="Select Model:").pack(anchor=tk.W, padx=10)
model_selector = ttk.Combobox(root, values=[
    "KNN", "SVM", "NB", "RF","LSTM"
])
model_selector.current(0)
model_selector.pack(fill=tk.X, padx=10, pady=(0, 10))

process_button = ttk.Button(root, text="Process", command=process_input)
process_button.pack(pady=5)

ttk.Label(root, text="Output:").pack(anchor=tk.W, padx=10, pady=(10, 0))
output_box = tk.Text(root, height=6, wrap=tk.WORD, state=tk.DISABLED)
output_box.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

root.mainloop()
