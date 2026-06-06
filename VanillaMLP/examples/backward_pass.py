import numpy as np

def backward(X, y_true, A1, A2, W2):
    batch_size = X.shape[0] # fetches number of samples

    # Output Layer gradients
    # gradient of cross-entropy + softmax combined (works out cleanly)
    dZ2 = A2.copy() # create a copy of the predicted probabilities (stored in A2 after softmax)
    dZ2[np.arange(batch_size), y_true] -= 1 # subtract 1 from correct class
    # np.arange(batch_size) creates row indices (when batch_size = 4, [0, 1, 2, 3])
    # if y_true labels are [0, 1, 0, 1]
    # dz2[np.arange(batch_size), y_true] creates:
        # dZ2[[0, 1, 2, 3], [0, 1, 0, 1]]
    # this shows which sample belongs to which label:
        # (0, 0)
        # (1, 1)
        # (2, 0)
        # (3, 1)
    # this will select the probabilities in those indexes in order to begin adjusting weights in a way that minimizes loss
    # -=1 will change the correct class entries to show a - (which means we need to INCREASE the probability of this neuron) or 
        # a + (which means we need to DECREASE the probability of the neuron)
    dZ2 /= batch_size       # average over batch
    # important to divide by batch_size so we get the average error per sample

    dW2 = A1.T @ dZ2        # (16, 2) - how much to adjust W2 based on value from
    # dW2 stores the gradients for the second layer of weights, and tells the network how much each individual weight
        # contributed to the final error
    # A1 holds the activity level of hidden neurons
    # dZ2 holds the final errors
    # A1.T @ dZ2 matrixx multiplication pairs up every single incoming activation with every single output error
    # Formula: Weight Adjustment (dW2) = Incoming Activation (A1)* Output Error
        # if a hidden neuron was 0, multiplying it by the error results in 0
        # if the neuron was large, multiplying by the error results in a large adjustment
    
    db2 = dZ2.sum(axis=0)   # (2,)  - how much to adjsut b2
    # the gradient of the bias is just the error of the neuron (dZ2) itself
    # equation for a single neuron is: Output = (Inputs * Weights) + Bias
    # the bias is the same for all inputs, and since we divided the error of dZ2 by the batch size, we can
        # sum up the collective bias to determine how wrong the bias itself was

    # At this point, the output layer adjustments have been figured out, so we step back into the hidden layer

    # hidden layer gradients
    dA1 = dZ2 @ W2.T        # (batch, 16) - error flowing back through W2
    # before calculating  how to fix the first layer of weights, we need to see how much total error was sent to the hidden
        # layer of activations
    # take the output error dZ2 and project it backward through the weights of the second layer (W2.T)
    # this "distributes the blame": if a hidden neuron was connected to an output neuron via a massive weight, and that
        # output neuron made a huge mistake, that hidden neuron bears a lot of the responsibility (dA1)
    # Explanation overall: during the forward pass, a hidden neuron sends its value (A1) forward through a weight (W2) to hit
        # the output neuron
    # during the backward pass, we just reverse that exact pipeline:
        # we take the output error (dZ2)
        # we multiply it by the weight (W2) that connected them
        # this gives us the error of the hidden neuron
    dZ1 = dA1 * (A1 > 0)    # (batch, 16) - ReLU gradient: 0 where A1 was 0
    # dZ1 is the error before the activation function
    # this is checking whether a neuron was firing and could contribute to the error or not
    # (A1 > 0): looks at actual outputs from the forward pass, and creates a matrix of True and False values
        # if a neuron was active (positive), it becomes True (which is treated as 1)
        # if a neuron was inactive (0), it becomes False, treated as 0)
    # the * Operator: element-wise multiplication (not matrix multiplication). It multiplies the raw blame (dA1) directly
        # against those 1's and 0's
        # if it multiplies by 1 (Active neuron) -> the error passes through completely untouched
        # if it multiplies by 0 (inactive neuron) -> the error becomes 0, and the blame is killed
    # the resulting matrix (dZ1) is the finalized error for the hidden layer, filtered through the reality of which neurons
        # were actually awake and participating

    dW1 = X.T @ dZ1         # (8, 16) - how much to adjust W1
    # multiplies the original raw inputs (X.T) by the hidden layer to see how to fix W1
    db1 = dZ1.sum(axis=0)   # (16,) - how much to adjust b1
    # sums up the hidden errors across the batch to get the single average update for the hidden biases b1

    return dW1, db1, dW2, db2