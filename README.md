# Transformer 2
Radi Akbar\
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
The most popular architectural update is the replacement of the Additive Positional Embedding (APE) with Rotary Positional Embedding (ROPE). While APE injects positional information by adding a positional embedding vector, ROPE rotates the embedding dimension of the by a certain angle to incorporate positional information. The ROPE technique has been shown to handle long-term dependency of relative positional information between words better than prior methods.

### SwiGLU
Another popular architectural update is the use of SwiGLU. Unlike RELU or GELU, SwiGLU takes in a projected vector in split it into two, afterwards it performs SiLU on the first chunk while multiplying it with the second chunk. The technique has been shown to improve performance hence why its adoption.

### RMSNorm
RMSNorm has recently been used more often in LLM papers. While the original LayerNorm seeks to prevent covariance shifts during training, RMSNorm reduces the number of computation by dividing the vector by its standard deviation to ensure invariance in the layer.

## Training
### Optimizer and Learning Rate Schedule
For this project, I follow LLAMA's and GPT Neo X's optimizer setup by using AdamW with $\beta_1 = 0.9, beta_2 = 0.95$ and a weight decay of 0.1 during training. In addition, I follow their cosine learning rate decay schedule with linear warmup as it's been use lately for training decoder-only and encoder-decoder architectures.

### Steps and Gradient Accumulation
The original paper created batches that each contain roughly 25k source and target tokens. According to this MediumDigest post, the author estimated that the mean batch size of the paper is 724 and that it ran for approximately 16 epochs. Due to resource limit, I'll be running it for 80k steps instead and set a batch size of 8. It's showed that higher batch sizes can help the model converge faster, as a solution I've implemented gradient accumulation to simulate a batch size of 724. Although in total I have ~9m steps to go over (562k for each epoch and there are 16 epochs) I will be only performing a gradient step every 90 step. \\
It also goes without saying that I clip the gradient of the model between [-1, 1] as it has become the standard practice for training LLMs.

### Configuration
TBD - But I'm thinking of renting a cluster of GPUs from Lambda Labs and train my model across nodes. Since I can't get a setup similar to the original "Attention is All You Need" paper, I've decided to just train it for longer but on a smaller setup.
