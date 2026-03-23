import numpy as np
from collections import Counter


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


def create_paires(dataset_lines, w2i, window=2):
    pairs = []
    for line in dataset_lines:
        words = line.split()
        # filter unknown words in one pass
        indices = [w2i[w] for w in words if w in w2i]
        n = len(indices)
        if n < 2:
            continue
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            # slice directly — no offset loop, no if checks
            context = indices[lo:i] + indices[i+1:hi]
            if context:
                pairs.append((context, indices[i]))
    return pairs

def initialize_weights(V, D, seed=123):
    rng = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D))
    W_out = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D)) 
    return W_in, W_out

def forward(W_in, W_out, context_indices):
    context_vecs = W_in[context_indices]
    h = np.mean(context_vecs, axis=0)
    scores = np.dot(W_out, h)
    
    e = np.exp(scores - np.max(scores))
    y_hat = e / np.sum(e)           
    
    return h, y_hat

def loss(y_hat, target_idx):
    return -np.log(y_hat[target_idx] + 1e-10)


def backward(W_in, W_out, context_indices, target_idx, h, y_hat, lr=0.01):

    d_scores = y_hat.copy()
    d_scores[target_idx] -= 1
    
    d_W_out = np.outer(d_scores, h)
    d_h = np.dot(W_out.T, d_scores)
    d_W_in = d_h / len(context_indices)

    W_out -= lr * d_W_out
    for idx in context_indices:
        W_in[idx] -= lr * d_W_in
    return W_in, W_out

def train(V, D, seed, dataset_lines, w2i, window_size, noisedist = None, epochs=100, lr=0.01, patience=3, min_delta=0.001):
    pairs = create_paires(dataset_lines, w2i, window_size)
    W_in, W_out = initialize_weights(V, D, seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]
    
    indices = np.arange(len(train_pairs))
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        total_loss = 0.0
        
        for i in indices:
            context_indices, target_idx = train_pairs[i]
            h, y_hat = forward(W_in, W_out, context_indices)
            total_loss += loss(y_hat, target_idx)
            W_in, W_out = backward(W_in, W_out, context_indices, target_idx, h, y_hat, lr)
        
        val_loss = sum(loss(forward(W_in, W_out, ctx)[1], tgt) for ctx, tgt in val_pairs) / len(val_pairs)

        train_loss = total_loss / len(train_pairs)
        print("epoch", epoch, ":  train=", train_loss, "  val=", val_loss)

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping at epoch", epoch, "— best val=", best_val_loss)
                break

    return W_in, W_out

import os
from datetime import datetime

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