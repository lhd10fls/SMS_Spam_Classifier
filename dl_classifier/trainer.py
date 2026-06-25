import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_fn(model, data_loader, optimizer, criterion, clip, device):
    from tqdm.auto import tqdm
    model.train()
    epoch_loss = 0

    for batch in tqdm(data_loader, desc="Training Batches", leave=False):
        x_train = batch[0].to(device)
        y_train = batch[1].to(device)

        optimizer.zero_grad()
        y_preds = model(x_train)

        loss = criterion(y_preds, y_train)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    from tqdm.auto import tqdm
    model.eval()
    epoch_loss = 0
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating Batches", leave=False):
            x_eval = batch[0].to(device)
            y_eval = batch[1].to(device)

            y_pred = model(x_eval)
            loss = criterion(y_pred, y_eval)

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def predict(data, model, threshold=0.5, device='cpu', data_type='train', pad_idx=1, max_len=200):
    model.eval()
    y_pred = []
    y_real = []
    
    with torch.inference_mode():
        for instance in data:
            x = instance['ids'][:max_len]
            if len(x) == 0:
                x = [pad_idx] * 10
    
            x = torch.LongTensor(x).unsqueeze(0).to(device)
            y = instance['label'] # Fixed KeyError: 'target' -> 'label'
            
            output_logits = model(x).item()
            pred_prob = torch.sigmoid(torch.tensor(output_logits)).item()
            pred = 1 if pred_prob > threshold else 0
            
            y_pred.append(pred)
            y_real.append(y)

    report = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"=== {data_type} dataset ===")
    print(f"F1 score: {f1:.4f}")
    print(f"Precision score: {precision:.4f}")
    print(f"Recall score: {recall:.4f}")
    print(f"Accuracy score: {accuracy_score(y_real, y_pred):.4f}")
    print(f"Balanced accuracy score: {balanced_accuracy_score(y_real, y_pred):.4f}\n")
    
    return y_pred, y_real
