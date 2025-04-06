import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import load_tokenizer, load_metadata

def predict_next_words(model, tokenizer, max_sequence_len, seed_text, next_words=3):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

def main():
    try:
        # Check if model files exist
        model_path = '../models/next_word_model.h5'
        tokenizer_path = '../models/tokenizer.pkl'
        metadata_path = '../models/metadata.json'
        
        if not all(os.path.exists(path) for path in [model_path, tokenizer_path, metadata_path]):
            print("Model assets not found. Please train the model first:")
            print("python src/model.py")
            return

        # Load assets
        model = load_model(model_path)
        tokenizer = load_tokenizer()
        metadata = load_metadata()
        
        if None in [model, tokenizer, metadata]:
            print("Failed to load model assets. Please retrain the model.")
            return
            
        max_sequence_len = metadata['max_sequence_len']
        
        # Get user input
        seed_text = input("Enter seed text: ")
        prediction = predict_next_words(model, tokenizer, max_sequence_len, seed_text)
        
        print(f"Predicted text: {prediction}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os  # Add this import at the top if not already present
    main()