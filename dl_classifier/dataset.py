import torch
import torch.nn as nn
from datasets import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX, PAD_IDX = 0, 1
SPECIAL_SYMBOLS = ['<unk>', '<pad>']

def prepare_hf_dataset(df_train, df_val, df_test):
    # Match old function structure: extract x and y
    train_data = [{"text": str(row['text']), "label": int(row['label'])} for _, row in df_train.iterrows()]
    val_data = [{"text": str(row['text']), "label": int(row['label'])} for _, row in df_val.iterrows()]
    test_data = [{"text": str(row['text']), "label": int(row['label'])} for _, row in df_test.iterrows()]

    # Convert to Hugging Face Dataset objects
    import pandas as pd
    train_data = Dataset.from_pandas(pd.DataFrame(data=train_data))
    val_data = Dataset.from_pandas(pd.DataFrame(data=val_data))
    test_data = Dataset.from_pandas(pd.DataFrame(data=test_data))
    
    return train_data, val_data, test_data

def tokenize_example(example, token_transform):
    tokens = token_transform(example['text'])
    return {"tokens": tokens}

def numericalize_example(example, vocab):
    ids = torch.LongTensor(vocab.lookup_indices(example['tokens']))
    return {"ids": ids}

def build_vocabulary(train_dataset, min_freq=2):
    vocab = build_vocab_from_iterator(
        train_dataset['tokens'],
        min_freq=min_freq,
        specials=SPECIAL_SYMBOLS,
        special_first=True
    )
    vocab.set_default_index(UNK_IDX)
    return vocab

def get_collate_fn(pad_idx=PAD_IDX, max_len=200):
    def collate_fn(batch):
        x_batch = [torch.tensor(sample['ids'][:max_len]) for sample in batch]
        y_batch = torch.tensor([sample['label'] for sample in batch], dtype=torch.float)

        # Padding for each batch
        x_batch = nn.utils.rnn.pad_sequence(x_batch, padding_value=pad_idx, batch_first=True)
        x_batch = x_batch.long()

        return x_batch, y_batch
    return collate_fn
