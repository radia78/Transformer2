# Transformer 2
A quality of life update to the architecture that we know and love! The world of Natural Language Processing is developing at a rampant speed, this simple sequence to sequence model is used to introduce some of the major updates to the transformer architecture.

# Architecture
The original architecture of the transformer from the "Attention is All You Need" paper includes a positional embedding layer for the source and target inputs, an encoder block that has a self-Attention mechanism and a feed-forward network, a decoder block that has causal Self-Attention and the Encoder block, and finally a Linear + Softmax layer for the outputs.

First, the position embedding layer that imbues the source and target texts with the positional information of the tokens. Second, is the encoder block that consists of a self-attention mechanism and a feed-forward network that later passes onto the decoder block. Third, the decoder block which consists of a self-attention mechanism, a cross-attention mechanism, and a feed-forward network before outputing a tensor of logits.

<img src="https://github.com/radia78/Transformer2/blob/main/images/transformer_architecture.png" alt="Original Architecture" width="321" height="435" align="center"/>


## Updates
### RoPE - Rotary Positional Embedding
One of the biggest updates to the transformer architecture is the use of rotary positional embedding (RoPE) to replace the original absolute positional embedding layer (APE). APE additiviely injects the position of each token by creating a vector where each odd and even element are populated by the values below.

$$f_{t \in \{k, q, v\}}(x_i, i) := W_{t \in \{q, k ,v\}}(x_i + p_i)$$
$$p_{i, 2t} = sin(k/10000^{2t/d}), p_{i, 2t + 1} = cos(k/10000^{2t/d})$$

Su et al.(2021) proposes a multiplicative method instead of an additive one through RoPE by rotating unit representations based on their position within a sequence.

<img src="https://github.com/radia78/Transformer2/blob/main/images/rope_example.png" alt="Original Architecture" width="638" height="435" align="center"/>

The authors find that RoPE slightly outperforms the vanilla transformers in the same machine translation task, but also handles long sequences better than APE, which makes it optimal for pre-training tasks and long-text generation.

### GELU - Gaussian Error Linear Unit
Another update to the transformer architecture is the replacement of ReLUs (Rectified Linear Unit) with GELUs (Gaussian Error Linear Unit). The ReLU replaced sigmoid activation functions in intermediate layers due to its ease of computation and non-vanishing gradients for large $x$ values.

$$ReLU(x) = max(x, 0)$$

However, the ReLU function cannot be differentiated at $x=0$. GELU solves the problem by making it differentiable at that point so the network can learn more complex patterns during training.

$$GELU(x) = x * \Phi (x)$$

Where $\Phi(x)$ is the cumulative distribution for the standard gaussian distribution. However, this is a computationally expensive, and so GELU is approximated to

$$GELU(x) \approx 0.5 * x * (1 + Tanh((2/\pi) * (x+0.044715∗x^3)))$$

### Flash Attention
One of the other updates that comes with PyTorch 2.0 is the introduction of flash-attention. Traditionally, the PyTorch transformers' attentions are calculated using a form of matrix multiplication, however this is computationally expensive and memory hungry. Dao et al.(2022) proposes IO awareness to the attention mechanism, making the transformers much more efficient.

# Training
TBA
