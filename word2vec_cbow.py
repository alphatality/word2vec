import numpy as np
import utils as init


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
            context = indices[lo:i] + indices[i+1:hi]
            if context:
                pairs.append((context, indices[i]))
    return pairs

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
    W_in, W_out = init.initialize_weights(V, D, seed)
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