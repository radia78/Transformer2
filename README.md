# Transformer 2
Radi Akbar
Personal Project

## Introduction
The world of NLP is developing at a rapid space. There has been so many updates to the original Transformer architecture, that its hard keeping track of all the progress that happened. This project showcases some of the recent developments in the Transformer architecture through tweaking the sequence-to-sequence model and compare its performance on English to German machine translation task.

## Dependencies
* Python 3.9.16
* torch 
* tqdm
* einops

## Architectural Updates
### Rotary Positional Embedding
The most popular architectural update is the replacement of the Additive Positional Embedding (APE) with Rotary Positional Embedding (ROPE). While APE injects positional information by adding a positional embedding vector, ROPE rotates the embedding dimension of the by a certain angle to incorporate positional information. You can read more about the specifics of ROPE and APE in Su, et al, 2021.

### SwiGLU
Another popular architectural update is the use of SwiGLU. Unlike RELU or GELU, SwiGLU takes in a projected vector in split it into two, afterwards it performs SiLU on the first chunk while multiplying it with the second chunk.

### RMSNorm
RMSNorm has recently been used more often in LLM papers. While the original LayerNorm seeks to prevent covariance shifts during training, RMSNorm reduces the number of computation by dividing the vector by its standard deviation to ensure invariance in the layer.

## Training
### Optimizer and Learning Rate Schedule
For this project, I follow LLAMA's and GPT Neo X's optimizer setup with $\beta_1 = 0.9, beta_2 = 0.95$ and implement a 0.1 weight decay during training.
In addition, I follow their cosine learning rate decay schedule with linear warmup as it's been use lately for training decoder-only and encoder-decoder architectures.

### Steps and Gradient Accumulation
Due to resource limitations, I cannot squeeze original paper's number of batches, instead I use a batch size of 4 and implement gradient accumulation with a step of 32, which simulates having a batch size of 128. It's been known that larger batch sizes help the model converge and gradient accumulation is the poorman's solution to the problem. Overall, I trained my model for roughly ~3 million steps, but because I update the gradient only every 32 step, the effective number of steps per epoch is roughly ~35k.

### Configuration
TBD - But I'm thinking of renting a cluster of GPUs from Lambda Labs and train my model across nodes. Since I can't get a setup similar to the original "Attention is All You Need" paper, I've decided to just train it for longer but on a smaller setup.
