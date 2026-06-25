import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout_rate=0.3):
        super(LSTM_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        self.lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim, 
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0, # Dropout is only applied between LSTM layers
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x has shape (batch_size, max_len)
        embedding = self.dropout(self.embedding(x))

        # lstm_output has shape (batch_size, max_len, hidden_dim)
        lstm_output, hidden = self.lstm(embedding)

        # Take output of the last LSTM cell
        # shape: (batch_size, 1, hidden_dim) -> squeeze gives (batch_size, hidden_dim)
        output_logits = self.fc(lstm_output[:, -1, :])

        # Return shape: (batch_size)
        return output_logits.squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
