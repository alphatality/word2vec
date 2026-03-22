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
        indices = [w2i[w] for w in words]
        n = len(indices)
        for i in range(n):
            context = []
            for offset in range(-window, window + 1):
                j = i + offset
                if offset == 0 or j < 0 or j >= n:
                    continue
                context.append(indices[j])
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

    y_hat[target_idx] -= 1
    
    d_W_out = np.outer(y_hat, h)
    
    # 3. Gradient par rapport à h
    #    → W_out.T multiplié par d_scores  →  shape (D,)
    d_h = np.dot(W_out.T, y_hat)
    
    # 4. Gradient par rapport à chaque ligne de W_in (contexte)
    #    → d_h divisé par le nombre de mots de contexte
    d_W_in = d_h / len(context_indices)
    
    # 5. Mise à jour SGD
    W_out -= lr * d_W_out
    for idx in context_indices:
        W_in[idx] -= lr * d_W_in
    return W_in, W_out

def train(W_in, W_out, seed, pairs, epochs=100, lr=0.01, patience=3, min_delta=0.001):
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
        
        val_loss = sum(
            loss(forward(W_in, W_out, ctx)[1], tgt)
            for ctx, tgt in val_pairs
        ) / len(val_pairs)

        train_loss = total_loss / len(train_pairs)
        print(f"Époque {epoch:>4d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping à l'époque {epoch} — meilleure val={best_val_loss:.4f}")
                break

    return W_in, W_out