import random
from pathlib import Path

import pandas as pd

methode = "cbow"  # or "skipgram"
if methode == "cbow":
    import word2vec_cbow as word2vec
else :
    import word2vec_skipgram as word2vec

D = 20         
LEARNING_RATE = 0.05
EPOCHS = 200
SEED = 257
window_size = 2
file_name = "dataset_wikipedia.parquet" #put it in data folder


BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / file_name
df = pd.read_parquet(file_path)
dataset = df["text"].astype(str).str.lower().tolist()
print(dataset[:5])



"""with open(file_path, "r") as f:
    dataset = f.read()
dataset_lines = dataset.strip().lower().split("\n")"""

dataset_lines = word2vec.prep_dataset(dataset)
V, w2i, i2w = word2vec.vocabulary(dataset_lines)
pairs = word2vec.create_paires(dataset_lines, w2i, window_size)

W_in, W_out = word2vec.initialize_weights(V, D, SEED)

W_in, W_out = word2vec.train(W_in, W_out, SEED, pairs, epochs=EPOCHS, lr=LEARNING_RATE)