import requests
import os

def download_dataset():
    DATA_URL = "https://www.gutenberg.org/files/48320/48320-0.txt"
    SAVE_DIR = "../data"
    SAVE_PATH = os.path.join(SAVE_DIR, "sherlock-holm.es_stories_plain-text_advs.txt")
    
    print("Downloading Sherlock Holmes dataset...")
    
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Dataset successfully saved to {SAVE_PATH}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()