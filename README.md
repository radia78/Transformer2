# Transformer 2
A quality of life update to the architecture that we know and love! The world of Natural Language Processing is developing at a rampant speed, this simple sequence to sequence model is used to introduce some of the major updates to the transformer architecture.

# Architecture
The original architecture of the transformer from the "Attention is All You Need" paper includes a Positional Embedding layer for the source and target inputs, an Encoder block that has a Self-Attention mechanism and an MLP, a Decoder block that has causal Self-Attention and the Encoder block, and finally a Linear + Softmax layer for the outputs.

<img src="https://github.com/radia78/Transformer2/blob/main/images/transformer_architecture.png" alt="Original Architecture" width="214" height="290"/>


## Updates
### RoPE - Rotary Positional Embedding
One of the biggest updates to the transformer architecture is the use of RoPE to replace the original positional embedding layer (Absolute Positional Embedding - APE). APE additiviely injects the position of each tokens while RoPE imbues the position of each token by rotating the embedding dimension.

$$f_{t \in \{k, q, v\}}(x_i, i) := W_{t \in \{q, k ,v\}}(x_i + p_i)$$

$$\begin{cases} p_{i, 2t} = sin(k/10000^{2t/d}), p_{i, 2t + 1} = cos(k/10000^{2t/d})\end{cases}$$


<img src="https://github.com/radia78/Transformer2/blob/main/images/rope_example.png" alt="Original Architecture"/>

### GELU - Gaussian Error Linear Unit
Another update to the transformer architecture is the replacement of ReLUs (Rectified Linear Unit) with GELUs. The GELU differs from the ReLU in a sense that the function is "smoother" near 0, which makes it possible to have gradients in negative ranges.

# Training
TBA
