# Transformer 2
A quality of life update to the architecture that we know and love! The world of Natural Language Processing is developing at a rampant speed, this simple sequence to sequence model is used to introduce some of the major updates to the transformer architecture.

# Architecture
The original architecture of the transformer from the "Attention is All You Need" paper includes a Positional Embedding layer for the source and target inputs, an Encoder block that has a Self-Attention mechanism and an MLP, a Decoder block that has causal Self-Attention and the Encoder block, and finally a Linear + Softmax layer for the outputs.

<img src="https://github.com/radia78/Transformer2/blob/main/images/transformer_architecture.png" alt="Original Architecture" width="428" height="580"/>


## Updates
### RoPE - Rotary Positional Embedding
One of the biggest updates to the transformer architecture is the use of RoPE to replace the original positional embedding layer (Absolute Positional Embedding - APE). APE additiviely injects the position of each tokens by creating an vector where each odd and even element are populated by the values below.

$$f_{t \in \{k, q, v\}}(x_i, i) := W_{t \in \{q, k ,v\}}(x_i + p_i)$$
$$p_{i, 2t} = sin(k/10000^{2t/d}), p_{i, 2t + 1} = cos(k/10000^{2t/d})$$

Su et al.(2021) proposes a multiplicative method instead of an additive one through RoPE by rotating unit representations based on their position within a sequence.

<img src="https://github.com/radia78/Transformer2/blob/main/images/rope_example.png" alt="Original Architecture" width="425" height="290"/>

The authors find that RoPE slightly outperforms the vanilla transformers in the same machine translation tasks, but also that RoPE handles long sequences better than APE, which makes it optimal for pre-training tasks and long text generation.

### GELU - Gaussian Error Linear Unit
Another update to the transformer architecture is the replacement of ReLUs (Rectified Linear Unit) with GELUs. The ReLU replaced sigmoid activation functions in intermediate layers due to its ease of computation and non-vanishing gradients at large $x$ values.

$$ReLU(x) = max(x, 0)$$

However, the ReLU function cannot be differentiated at $x=0$. GeLU fixes this problem by making it differentiable at that point so the network can learn more complex patterns.

$$GeLU(x) = x * \Phi (x)$$

Where $\Phi(x)$ is the cumulative distribution for the Gaussian distribution with $\mu=1, \sigma^2=0$. However, this is a bit computationally expensive, and so GeLU has always been approximated to

$$GeLU(x) \approx 0.5 * x * (1 + Tanh((2/\pi) * (x+0.044715∗x^3)))$$

### Flash Attention
One of the other updates that comes with PyTorch 2.0 is the introduction of flash-attention. Traditionally, the PyTorch transformers' attentions are calculated using a form of matrix multiplication, however this is computationally expensive and memory hungry. Dao et al.(2022) proposes IO awareness to the attention mechanism, making the transformers much more efficient than before.

# Training
TBA
