import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Text Preprocessing Function (same as before) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# --- PyTorch Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()
        # Convert tokens to indices
        indexed = [self.word_to_idx.get(word, 1) for word in tokens] # 1 is the index for <UNK>
        # Pad sequence
        if len(indexed) < self.max_len:
            indexed += [0] * (self.max_len - len(indexed)) # 0 is the index for <PAD>
        else:
            indexed = indexed[:self.max_len]
        
        label = self.labels[idx]
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# --- PyTorch Model Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Get the output of the last time step
        final_output = lstm_out[:, -1, :]
        out = self.fc(final_output)
        return self.sigmoid(out)

# --- Training Script ---
# 1. Load and Prepare Data
print("Loading and preparing data...")
df = pd.read_csv('train.csv')
# df = df.sample(n=30000, random_state=42) # Use a smaller sample for faster training
df['comment_text'] = df['comment_text'].astype(str)
df['cleaned_text'] = df['comment_text'].apply(clean_text)

# 2. Build Vocabulary
print("Building vocabulary...")
all_text = ' '.join(df['cleaned_text'])
word_counts = Counter(all_text.split())
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# Vocab mapping
MAX_VOCAB_SIZE = 20000
vocab_to_int = {word: i+2 for i, word in enumerate(sorted_vocab) if i < MAX_VOCAB_SIZE - 2}
vocab_to_int['<PAD>'] = 0
vocab_to_int['<UNK>'] = 1
print(f"Vocabulary size: {len(vocab_to_int)}")

# 3. Create DataLoaders
X = df['cleaned_text'].tolist()
y = df['toxic'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MAX_LEN = 150
train_dataset = TextDataset(X_train, y_train, vocab_to_int, MAX_LEN)
test_dataset = TextDataset(X_test, y_test, vocab_to_int, MAX_LEN)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 4. Instantiate the model, loss function, and optimizer
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
OUTPUT_DIM = 1
N_LAYERS = 1
DROPOUT = 0.3

model = LSTMClassifier(len(vocab_to_int), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("Starting training...")
EPOCHS = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 6. Save Model and Vocabulary
print("Saving model and vocabulary...")
model_info = {
    "model_state_dict": model.state_dict(),
    "vocab_to_int": vocab_to_int,
    "max_len": MAX_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "n_layers": N_LAYERS,
    "dropout": DROPOUT
}
torch.save(model_info, 'toxicity_model.pth')
print("\n--- Training complete! Model saved as toxicity_model.pth ---")