import numpy as np

def cross_entropy_loss(probs, y_true):
    # probs: (batch, n_classes) - softmax output
    # y_true: (batch,) - integer class labels e.g. [0, 1, 1, 0]

    batch_size = probs.shape[0]
    
    # grab the probability assigned to the correct class for each sample
    correct_probs = probs[np.arange(batch_size), y_true]

    # -log of each, then average
    loss = -np.mean(np.log(correct_probs + 1e-8)) # 1e-8 prevents log(0)
    return loss

# test it
y_fake = np.array([1, 0, 1, 0]) # fake labels for our 4 samples
loss = cross_entropy_loss(probs, y_fake)
print(f"Loss: {loss:.4f}")