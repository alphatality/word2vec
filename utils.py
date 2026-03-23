from collections import Counter
import numpy as np 
import os
from datetime import datetime

def prep_dataset(dataset):
    dataset_lines = dataset.strip().lower().split("\n")
    return dataset_lines

def vocabulary(dataset_lines):
    dataset_words = [i.split() for i in dataset_lines]
    dataset_words = [word for line in dataset_words for word in line]
    words = Counter(dataset_words)
    vocab = [word for word, _ in words.most_common()]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return len(words), w2i, i2w


def initialize_weights(V, D, seed=123):
    rng = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D))
    W_out = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D)) 
    return W_in, W_out

def save_model(W_in, W_out, w2i, i2w, base_path="models"):
    os.makedirs(base_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_path, timestamp)
    os.makedirs(model_dir)
    
    np.save(os.path.join(model_dir, "W_in.npy"), W_in)
    np.save(os.path.join(model_dir, "W_out.npy"), W_out)
    np.save(os.path.join(model_dir, "w2i.npy"), w2i)
    np.save(os.path.join(model_dir, "i2w.npy"), i2w)
    
    print(f"Model saved → {model_dir}")
    return model_dir  

def load_model(path):
    W_in  = np.load(os.path.join(path, "W_in.npy"))
    W_out = np.load(os.path.join(path, "W_out.npy"))
    w2i   = np.load(os.path.join(path, "w2i.npy"),  allow_pickle=True).item()
    i2w   = np.load(os.path.join(path, "i2w.npy"),  allow_pickle=True).item()
    print(f"Model loaded — vocab={len(w2i)}, D={W_in.shape[1]}")
    return W_in, W_out, w2i, i2w

def plus_proches(W_in, w2i, i2w, mot, top_k=5):
    if mot not in w2i:
        print(f"'{mot}' no vocab")
        return []
    
    idx = w2i[mot]
    vecteur = W_in[idx]
    
    normes = np.linalg.norm(W_in, axis=1)             
    scores = W_in @ vecteur                          
    cosinus = scores / (normes * np.linalg.norm(vecteur) + 1e-10)  
    
    cosinus[idx] = -1
    
    top_indices = np.argsort(cosinus)[::-1][:top_k]
    
    return [(cosinus[i], i2w[i]) for i in top_indices]