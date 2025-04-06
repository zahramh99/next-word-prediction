import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import json
import os

def preprocess_data(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))
    
    return X, y, tokenizer, max_sequence_len, total_words

def build_model(total_words, max_sequence_len):
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(150),
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])
    return model

def train_model():
    # Check if data exists
    data_path = '../data/sherlock-holm.es_stories_plain-text_advs.txt'
    if not os.path.exists(data_path):
        print("Dataset not found. Please run:")
        print("python data/download_dataset.py")
        return
    
    # Load and process data
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
    
    # Preprocess
    X, y, tokenizer, max_sequence_len, total_words = preprocess_data(text)
    
    # Build and train model
    model = build_model(total_words, max_sequence_len)
    print("Training model... (This may take a while)")
    model.fit(X, y, epochs=100, verbose=1)
    
    # Save assets
    os.makedirs('../models', exist_ok=True)
    try:
        save_model(model, '../models/next_word_model.h5')
        
        with open('../models/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        metadata = {
            'max_sequence_len': max_sequence_len,
            'total_words': total_words
        }
        with open('../models/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print("Model training complete and assets saved to models/ directory")
    except Exception as e:
        print(f"Error saving model assets: {e}")

def load_tokenizer():
    try:
        with open('../models/tokenizer.pkl', 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def load_metadata():
    try:
        with open('../models/metadata.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

if __name__ == "__main__":
    train_model()