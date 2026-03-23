from pathlib import Path
import pandas as pd

methode = "skipgram"  # or "skipgram"
if methode == "cbow":
    import word2vec_cbow as word2vec
else :
    import word2vec_skipgram as word2vec

D = 9         
LEARNING_RATE = 0.05
EPOCHS = 20
SEED = 123
window_size = 2
patience = 5
min_delta = 0.0001
file_name_train = "train.txt" #put it in data folder, txt or parquet
file_name_test = "test.txt" #put it in data folder, txt or parquet

BASE_DIR = Path(__file__).resolve().parent
file_path_train = BASE_DIR / "data" / file_name_train
file_path_test = BASE_DIR / "data" / file_name_test

"""chunk_size=1000
df = pd.read_parquet(file_path)
dataset = df["text"].astype(str).str.lower().tolist() # uncomment if parquet, otherwise use the txt reading method below
words = dataset.split()
lines = [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]"""
with open(file_path_train, "r") as f:
        dataset_train = f.read()
lines_train = word2vec.prep_dataset(dataset_train)
with open(file_path_test, "r") as f:
        dataset_test = f.read()
lines_test = word2vec.prep_dataset(dataset_test)

def train_model(lines):

    V, w2i, i2w = word2vec.vocabulary(lines)

    W_in, W_out = word2vec.train(V,D, SEED,lines, w2i, window_size, epochs=EPOCHS, lr=LEARNING_RATE,patience=patience,min_delta=min_delta)

    return word2vec.save_model(W_in, W_out, w2i, i2w)
    

def test_model(model,lines_test):
    W_in, W_out, w2i, i2w = word2vec.load_model(model)
    from collections import Counter
    word_counts = Counter(w for line in lines_test for w in line.split())
    

    stopwords = {"the", "this", "that", "every"}
    candidates = [
        w for w, _ in word_counts.most_common(100)
        if w in w2i and w not in stopwords
    ][:20]

    print(f"Modèle : {model}")
    print(f"Vocab  : {len(w2i)} mots  D={W_in.shape[1]}")
    print("-" * 55)

    for mot in candidates:
        voisins = word2vec.plus_proches(W_in, w2i, i2w, mot, top_k=5)
        if voisins:
            print(f"{mot:20s} → {', '.join(f'{w}({s:.2f})' for s,w in voisins)}")

    print("-" * 55)
test_model(train_model(lines_train),lines_test)
