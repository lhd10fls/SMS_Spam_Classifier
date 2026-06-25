import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from pathlib import Path

# Import our refactored DL components
from dl_classifier.dataset import (
    prepare_hf_dataset, 
    tokenize_example, 
    numericalize_example, 
    build_vocabulary, 
    get_collate_fn,
    PAD_IDX
)
from dl_classifier.model import LSTM_model, count_parameters
from dl_classifier.trainer import set_seed, train_fn, evaluate_fn, predict

def main():
    # 1. Setup Environment
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up Paths (Cross-platform compatible)
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "Data" / "combined_spam_dataset.csv"
    MODEL_DIR = BASE_DIR / "Model" / "Lstm"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Split dataset precisely as before
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Convert to HuggingFace Datasets
    train_data, val_data, test_data = prepare_hf_dataset(train_df, val_df, test_df)

    # 3. Tokenization & Vocabulary
    print("Tokenizing data...")
    token_transform = get_tokenizer('basic_english')
    fn_kwargs_token = {'token_transform': token_transform}
    
    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs_token)
    val_data = val_data.map(tokenize_example, fn_kwargs=fn_kwargs_token)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs_token)

    # Building vocab
    print("Building vocabulary...")
    vocab = build_vocabulary(train_data, min_freq=2)
    print(f"Vocabulary size: {len(vocab)}")

    # Numericalize
    fn_kwargs_num = {'vocab': vocab}
    train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs_num)
    val_data = val_data.map(numericalize_example, fn_kwargs=fn_kwargs_num)
    test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs_num)

    # 4. Data Loaders
    BATCH_SIZE = 128
    collate_fn = get_collate_fn(PAD_IDX, max_len=200)

    train_data_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_data_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # 5. Initialize Model
    embedding_dim = 200
    hidden_dim = 200
    num_layers = 1

    model = LSTM_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    n_epochs = 10
    clip = 2
    best_val_loss = float('inf')
    train_loss_val = []
    val_loss_val = []

    model_path = MODEL_DIR / "LSTM_model_best.pth"

    print("Starting training...")
    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss = train_fn(model, train_data_loader, optimizer, criterion, clip, device)
        val_loss = evaluate_fn(model, val_data_loader, criterion, device)
        
        train_loss_val.append(train_loss)
        val_loss_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the state_dict instead of whole model
            torch.save(model.state_dict(), model_path)
            
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Train loss: {train_loss:7.3f}, Train perplexity: {np.exp(train_loss):7.3f}")
        print(f"Val loss:   {val_loss:7.3f}, Val perplexity:   {np.exp(val_loss):7.3f}")

    # 7. Save Vocabulary and Artifacts
    vocab_path = MODEL_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    # Also save state_dict of final epoch just in case
    final_model_path = MODEL_DIR / "LSTM_model_final.pth"
    torch.save(model.state_dict(), final_model_path)

    # Plotting Learning Curve
    plt.figure()
    plt.plot(range(1, n_epochs + 1), np.exp(train_loss_val), label='Train perplexity')
    plt.plot(range(1, n_epochs + 1), np.exp(val_loss_val), label='Validation perplexity')
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.title("Perplexity plot")
    plt.legend()
    plot_path = MODEL_DIR / "learning_curve.png"
    plt.savefig(plot_path)
    print(f"Saved learning curve to {plot_path}")

    # 8. Evaluation
    print("\n--- Final Evaluation ---")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    test_loss = evaluate_fn(model, test_data_loader, criterion, device)
    print(f"Test loss: {test_loss:7.3f}, Test perplexity: {np.exp(test_loss):7.3f}\n")

    predict(train_data, model, threshold=0.5, device=device, data_type='Train', pad_idx=PAD_IDX)
    predict(val_data, model, threshold=0.5, device=device, data_type='Validation', pad_idx=PAD_IDX)
    predict(test_data, model, threshold=0.5, device=device, data_type='Test', pad_idx=PAD_IDX)

if __name__ == "__main__":
    main()
