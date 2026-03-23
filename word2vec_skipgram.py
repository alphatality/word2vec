import numpy as np
from collections import Counter
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


def create_paires(dataset_lines, w2i, window=2):
    pairs = []
    for line in dataset_lines:
        words = line.split()
        indices = [w2i[w] for w in words if w in w2i]
        n = len(indices)
        if n < 2:
            continue
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((indices[i], indices[j])) 
    return pairs

def initialize_weights(V, D, seed=123):
    rng = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D))
    W_out = rng.uniform(-0.5 / D, 0.5 / D, size=(V, D)) 
    return W_in, W_out

def build_noise_distribution(w2i, dataset_lines, power=0.75):
    counts = np.zeros(len(w2i))
    for line in dataset_lines:
        for w in line.split():
            if w in w2i:
                counts[w2i[w]] += 1
    counts = counts ** power
    return counts / counts.sum()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def train_pair(W_in, W_out, center_idx, context_idx, noise_dist, k=5, lr=0.025):
    v_center = W_in[center_idx].copy()
    negatives = []
    while len(negatives) < k:
        sample = np.random.choice(len(noise_dist), p=noise_dist)
        if sample != context_idx:
            negatives.append(sample)
    v_pos    = W_out[context_idx].copy()
    sig_pos = sigmoid(np.dot(v_center, v_pos))
    loss    = -np.log(sig_pos + 1e-10)

    grad_center = (sig_pos - 1) * v_pos 
    for neg_idx in negatives:
        v_neg = W_out[neg_idx].copy()
        sig_neg = sigmoid(np.dot(v_center, v_neg))
        loss   += -np.log(1 - sig_neg + 1e-10)

        grad_center         += sig_neg * v_neg
        W_out[neg_idx]      -= lr * sig_neg * v_center

    W_out[context_idx] -= lr * (sig_pos - 1) * v_center
    W_in[center_idx]   -= lr * grad_center

    return loss

def train(V, D, seed, dataset_lines, w2i, window_size, epochs=100, lr=0.025, k=5, patience=3, min_delta=0.01):
    noise_dist = build_noise_distribution(w2i, dataset_lines)
    pairs =list(create_paires(dataset_lines, w2i, window_size))
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
            center_idx, context_idx = train_pairs[i]
            total_loss += train_pair(W_in, W_out, center_idx, context_idx, noise_dist, k, lr)
        val_loss = 0.0
        for center_idx, context_idx in val_pairs:
            sig = sigmoid(np.dot(W_in[center_idx], W_out[context_idx]))
            val_loss += -np.log(sig + 1e-10)
        val_loss /= len(val_pairs)

        train_loss = total_loss / len(train_pairs)
        print(f"Epoch {epoch:>4d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping — best val={best_val_loss:.4f}")
                break

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