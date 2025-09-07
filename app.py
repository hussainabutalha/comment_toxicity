import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn

# --- Preprocessing functions (must be identical to training) ---
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# --- PyTorch Model Definition (must be identical to training) ---
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
        final_output = lstm_out[:, -1, :]
        out = self.fc(final_output)
        return self.sigmoid(out)

# --- Load Model and Vocabulary ---
@st.cache_resource
def load_assets():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_info = torch.load('toxicity_model.pth', map_location=device)
        
        vocab_to_int = model_info['vocab_to_int']
        vocab_size = len(vocab_to_int)
        
        model = LSTMClassifier(
            vocab_size,
            model_info['embedding_dim'],
            model_info['hidden_dim'],
            model_info['output_dim'],
            model_info['n_layers'],
            model_info['dropout']
        )
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        model.eval() # Set model to evaluation mode
        
        return model, vocab_to_int, model_info['max_len'], device
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.error("Please make sure 'toxicity_model.pth' is in the same directory.")
        return None, None, None, None

model, vocab_to_int, max_len, device = load_assets()

# --- Prediction Function ---
def predict_toxicity(text):
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    indexed = [vocab_to_int.get(word, 1) for word in tokens] # 1 is <UNK>
    
    if len(indexed) < max_len:
        indexed += [0] * (max_len - len(indexed)) # 0 is <PAD>
    else:
        indexed = indexed[:max_len]
        
    tensor_input = torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_input)
        prediction = output.item()
    return prediction

# --- Streamlit App UI ---
st.set_page_config(page_title="Comment Toxicity Detector", layout="wide")
st.title("🧪 PyTorch for Comment Toxicity Detection")
st.markdown("An interactive web app to detect toxicity using a trained LSTM model with PyTorch.")

# Single Comment Prediction
st.header("Single Comment Analysis")
user_input = st.text_area("Enter a comment below:", "This is a sample comment.", height=150)

if st.button("Analyze Comment"):
    if model:
        if user_input:
            with st.spinner('Analyzing...'):
                prediction = predict_toxicity(user_input)
                toxicity_score = round(prediction * 100, 2)
                
                st.write(f"**Toxicity Score:** {toxicity_score}%")
                st.progress(prediction)
                
                if prediction > 0.5:
                    st.error("This comment is likely **TOXIC**.")
                else:
                    st.success("This comment is likely **NOT TOXIC**.")
        else:
            st.warning("Please enter a comment to analyze.")
    else:
        st.stop()