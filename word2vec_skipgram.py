import numpy as np
import utils



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
    W_in, W_out = utils.initialize_weights(V, D, seed)
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
