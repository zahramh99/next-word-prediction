# Next Word Prediction Model

This project implements a Next Word Prediction model using LSTM neural networks in TensorFlow/Keras.

## Description
Next Word Prediction means predicting the most likely word or phrase that will come next in a sentence or text. It's used in applications like messaging apps, search engines, and virtual assistants.

## Features
- LSTM-based neural network architecture
- Trained on Sherlock Holmes stories dataset
- Generates predictions based on input context

## Installation
1. Clone this repository
2. Install requirements:
```bash
pip install -r requirements.txt

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/yourusername/next-word-prediction.git
cd next-word-prediction
```

2. Download the dataset
```bash
python data/download_dataset.py
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run the model
```bash
python src/main.py
```
1. Download data: `python data/download_dataset.py`
2. Train model: `python src/model.py`
3. Make predictions: `python src/main.py`