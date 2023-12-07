# Neural Network Implementations from Scratch

This repository contains implementations of various neural networks from scratch. (without using ML and DL Libaries)

Each implementation is designed to provide a fundamental understanding of how these networks work. 

The implementations cover a range of neural network architectures and concepts.

## List of Implementations

1. **AutoEncoder.py**
    - Implements an autoencoder for feature learning and dimensionality reduction.
    - **Equation:**
      ```math
      z = sigmoid(W_encoder * x + b_encoder)
      x_hat = sigmoid(W_decoder * z + b_decoder)
      ```

2. **BoltzmannMachine.py**
    - Basic implementation of a Boltzmann Machine for unsupervised learning.
    - **Equation:**
      ```math
      P(v, h) = exp(-E(v, h)) / Z
      ```

3. **GenerativeAdversarialNetwork.py**
    - Simple implementation of a Generative Adversarial Network (GAN) with a basic generator and discriminator.
    - **Equation:**
      ```math
      Loss = log(D(x)) + log(1 - D(G(z)))
      ```

4. **HopfieldNetwork.py**
    - Implementation of a Hopfield Network, a content-addressable memory system.
    - **Equation:**
      ```math
      H(x) = -0.5 * x.T * W * x
      ```

5. **LongShortTermMemoryLSTM.py**
    - Basic implementation of a Long Short-Term Memory (LSTM) network for handling sequential data.
    - **Equations:**
      ```math
      f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
      i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
      o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
      c_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
      h_t = o_t * tanh(c_t)
      ```

6. **Microsoft_Stock.csv**
    - Dataset file containing historical Microsoft stock data.

7. **MultiLayerPerceptronClassification.py**
    - Implementation of a multi-layer perceptron (MLP) for classification tasks.
    - **Equation (for a single neuron):**
      ```math
      y = activation(W * x + b)
      ```

8. **MultiLayerPerceptronRegression.py**
    - Implementation of an MLP for regression tasks.

9. **RadialBasisFunctionNetworks.py**
    - Implementation of a Radial Basis Function Network (RBFN) for function approximation.
    - **Equation:**
      ```math
      φ(x, c) = exp(-‖x - c‖² / (2 * σ²))
      ```

10. **SelfAttentionMechanism.py**
    - Implementation of a self-attention mechanism, commonly used in transformers.
    - **Equation:**
      ```math
      Attention(Q, K, V) = softmax(QK.T / sqrt(d_k))V
      ```

11. **SimpleCNN.py**
    - Basic Convolutional Neural Network (CNN) implementation.
    - **Equation (for convolution):**
      ```math
      Z[i, j] = ∑∑(X[i:i+f, j:j+f] * W) + b
      ```

12. **SimpleEncoderDecoder.py**
    - Simple implementation of an encoder-decoder architecture.

13. **SimpleRNN.py**
    - Implementation of a Simple Recurrent Neural Network (RNN).
    - **Equation:**
      ```math
      h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
      ```

14. **SingleLayerPerceptronClassification.py**
    - Implementation of a single-layer perceptron (SLP) for classification tasks.

15. **SingleLayerPerceptronRegression.py**
    - Implementation of an SLP for regression tasks.

16. **TitanicSurvialBySingleLayerPerceptron.py**
    - Example of using a single-layer perceptron to predict Titanic survival.

17. **Transformer.py**
    - Basic implementation of the Transformer architecture.

18. **kali.jpg**
    - Sample image for testing.

19. **positionalEncoding.py**
    - Implementation of positional encoding used in transformers.
    - **Equation:**
      ```math
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
      ```

20. **scaled_dot_product_attention.py**
    - Implementation of scaled dot-product attention used in transformers.
    - **Equation:**
      ```math
      Attention(Q, K, V) = softmax(QK.T / sqrt(d_k))V
      ```

21. **titanic.csv**
    - Dataset file containing Titanic passenger data.

## How to Use

Each Python script can be run independently. Ensure you have the required dependencies installed. You can run the scripts using a Python interpreter.

```bash
python AutoEncoder.py
```
