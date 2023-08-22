# Transformer 2
Radi Akbar <br>
Personal Project

## Introduction
The world of NLP is developing at a rapid space. There has been so many updates to the original Transformer architecture, that its hard keeping track of all the progress that happened. This project showcases some of the recent developments in the Transformer architecture through tweaking the sequence-to-sequence model and compare its performance on English to German machine translation task

## Training Specification
### Architecture
The architecture of the model remains the same from Vaswani, et al, 2017 but with a few modifications. First update is replacing the Additional Positional Embedding (APE) with Rotary Positional Embedding (ROPE) from Su, et al, 2021. ROPE rotates the embedded sequences to inject positional information instead of adding it. It's been shown that ROPE can hold long-term dependency between words in huge context windows. <br>

The second update is replacing LayerNorm with RMSNorm from Zhang & Sennrich, 2019. LayerNorm addresses the problem of covariance shifts during training by standardizing the inputs by its mean and standard deviation. RMSNorm simplifies the normalization process by only dividing the input by its standard deviation. <br>

The third update is replacing the ReLU activation function with SwiGLU. It's been shown by Shazeer, 2019 that SwiGLU improves the performance of NLP tasks. The paper also specifies that to make computation less expensive, they change the hidden size of the MLP component by multiplying it with 2/3.

### Training Setup
I follow LLAMA's and GPT Neo-X's training setup by using an AdamW optimizer with $\beta_1 = 0.9, \beta_2 = 0.95$ and a weight decay of 0.1. In addition, I follow their cosine learning rate decay schedule as its recently been used more often for training Encoder-Decoder and Decoder-Only architectures. <br>

Due to resource limitations, 

## Data & Tokenizer Preparation
## Training the Model
## Performance Evaluation
### BLEU Score
