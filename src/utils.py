import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_input(tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    return token_list

def decode_prediction(predicted, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""