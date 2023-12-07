import numpy as np

def scaled_dot_product_attention(query, key, value):
    """
    Scaled Dot-Product Attention mechanism.

    Parameters:
    - query (numpy.ndarray): Query vector.
    - key (numpy.ndarray): Key vector.
    - value (numpy.ndarray): Value vector.

    Returns:
    numpy.ndarray: Context vector.
    """
    d_k = query.shape[-1]  # Dimension of query/key vectors
    scores = np.dot(query, key.T) / np.sqrt(d_k)  # Dot product with scaling
    attention_weights = softmax(scores, axis=-1)  # Apply softmax to get attention weights
    context_vector = np.dot(attention_weights, value)  # Weighted sum to get context vector
    return context_vector

def softmax(x, axis=-1):
    """
    Softmax function.

    Parameters:
    - x (numpy.ndarray): Input array.
    - axis (int): Axis along which the softmax is computed.

    Returns:
    numpy.ndarray: Softmax output.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
