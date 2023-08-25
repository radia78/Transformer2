# Transformer 2
Radi Akbar <br>
Personal Project

## Introduction
The world of NLP is developing at a rapid space. There has been so many updates to the original Transformer architecture, that its hard keeping track of all the progress that happened. This project showcases some of the recent developments in the Transformer architecture through tweaking the sequence-to-sequence model and compare its performance on English to German machine translation task

## Training Specifications
### Architecture
The architecture of the model remains the same from Vaswani, et al, 2017 but with a few modifications. First update is replacing the Additional Positional Embedding (APE) with Rotary Positional Embedding (ROPE) from Su, et al, 2021. ROPE rotates the embedded sequences to inject positional information instead of adding it. It's been shown that ROPE can hold long-term dependency between words in huge context windows. <br>

The second update is replacing LayerNorm with RMSNorm from Zhang & Sennrich, 2019. LayerNorm addresses the problem of covariance shifts during training by standardizing the inputs by its mean and standard deviation. RMSNorm simplifies the normalization process by only dividing the input by its standard deviation. <br>

The third update is replacing the ReLU activation function with SwiGLU. It's been shown by Shazeer, 2019 that SwiGLU improves the performance of NLP tasks. The paper also specifies that to make computation less expensive, they change the hidden size of the MLP component by multiplying it with 2/3.

### Training Setup
I follow LLAMA's and GPT Neo-X's training setup by using an AdamW optimizer with $\beta_1 = 0.9, \beta_2 = 0.95$ and a weight decay of 0.1. In addition, I follow their cosine learning rate decay schedule as its recently been used more often for training Encoder-Decoder and Decoder-Only architectures. <br>

It's been shown on various papers that training on larger batch sizes help the model converge faster, but due to resource limitations I only train the model on batch size of 8. To address this problem, I used gradient accumulation so that the model only update its gradient every 32 step to simulate a batch size of 256. <br> 

According to this Medium Article, the original paper trains the model on 16 epochs with a batch size of 724. Since there's roughly 4.5m rows in the training dataset, that means it ran through approximately 6.2k steps per epoch and totaling to 100k steps. Due to my resource limitations, I'm only training my model for 80k steps, which approximates to running the model for 13 epochs. 

## Usage
### Dataset Preparation
To prepare the data I downloaded Stanford's WMT14 Dataset from [here](https://nlp.stanford.edu/projects/nmt/) and performed some text cleaning like converting "##UNDERSCORE##" to "_". Afterwards, I convert the dataset into a map style dataset and push it to Huggingface Hub. To use the dataset, use the following line of code:
```
from datasets import load_dataset
data = load_dataset('radia/wmt14-de2en')
```
The full dataset can be viewed in my HuggingFace page [here](https://huggingface.co/)

### Tokenizer Preparation
To prepare the tokenizer, I train a BPE tokenizer using my cleaned dataset. I made sure that the vocabulary size of the tokenizer is consistent with the one from the paper (roughly 37k tokens between German). To train the tokenizer, run the following code on your terminal:
'''
python3 prepare_tokenizers.py
'''

## Training the Model
## Performance Evaluation
## References
<a id="1">[1]</a> 
[Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.](https://arxiv.org/abs/1706.03762)

<a id="2">[2]</a> 
[Biao Zhang and Rico Sennrich. 2019. Root mean
square layer normalization.](https://arxiv.org/abs/1910.07467)

<a id="3">[3]</a> 
[Noam Shazeer. 2020. Glu variants improve transformer.](https://arxiv.org/abs/2002.05202)

<a id="4">[4]</a> 
[Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha,
Bo Wen, and Yunfeng Liu. 2021. Roformer: Enhanced transformer with rotary position embedding.](https://arxiv.org/abs/2104.09864)

<a id="5">[5]</a> 
[Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2021. GLM: General Language Model Pretraining with Autoregressive Blank Infilling.](https://arxiv.org/abs/2210.02414)

<a id="6">[6]</a> 
[Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models.](https://arxiv.org/abs/2302.13971)

- https://github.com/karpathy/nanoGPT/blob/master/model.py
