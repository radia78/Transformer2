import torch
from rope import *
import math
import logging
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

"""
Attention was inspired by Andrej Karpathy's MinGPT https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# implement the self attention mechanism for rotated embeddings
class CausalSelfAttention(nn.Module):
    def __init__(self, config, causal=False):
        super(CausalSelfAttention, self).__init__()
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.causal = causal

        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb) # convert the embedded text input into the Q, K V
        self.c_proj = nn.Linear(config.n_emb, config.n_emb) # output projection

        # regularization
        self.pos_dropout = nn.Dropout(config.dropout) # the dropout applied by the 
        self.attn_dropout = nn.Dropout(config.dropout) # the attention dropout
        self.res_dropout = nn.Dropout(config.dropout) # the residual dropout

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if causal:
            self.register_buffer("bias", torch.tril(
                torch.ones(config.max_len, config.max_len)).view(1, 1, config.max_len, config.max_len))

    def pos_encoding(self, x):
        cos, sin = Rotary(self.n_emb // self.n_head)(x)
        return self.pos_dropout(apply_rotary_pos_emb(x, cos, sin))

    def forward(self, x):
        B, T, C = x.size() # shape of tensor will always be (batch, seq_len, n_emb)

        # calculate Q, K, V for all heads and batch
        Q, K, V = self.c_attn(x).split(self.n_emb, dim=-1)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Pass ELU to the Q, K - per Eq. 19 from Sue, et al
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # applying the rotary positional embedding to Q and K
        Q_rot = self.pos_encoding(Q)
        K_rot = self.pos_encoding(K)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        denom = Q @ K.transpose(-2, -1)
        att = Q_rot @ K_rot.transpose(-2, -1)
        if self.causal:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att / denom.sum(dim=-1, keepdim=True)
        att = self.attn_dropout(att)
        y = att @ V # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, nh, hs)

        # output projection
        y = self.res_dropout(self.c_proj(y))
        return y
    
class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.q_attn = nn.Linear(config.n_emb, config.n_emb)
        self.kv_attn = nn.Linear(config.n_emb, 2 * config.n_emb) # convert the embedded text input into the K, V
        self.c_proj = nn.Linear(config.n_emb, config.n_emb) # output projection

        # regularization
        self.pos_dropout = nn.Dropout(config.dropout) # the dropout applied by the 
        self.attn_dropout = nn.Dropout(config.dropout) # the attention dropout
        self.res_dropout = nn.Dropout(config.dropout) # the residual dropout

    def pos_encoding(self, x):
        cos, sin = Rotary(self.n_emb // self.n_head)(x)
        return self.pos_dropout(apply_rotary_pos_emb(x, cos, sin))

    def forward(self, x, memory):
        B, S, C = memory.size() # shape of tensor will always be (batch, seq_len, n_emb)
        _, T, _ = x.size()

        # calculate Q, K, V for all heads and batch
        K, V = self.kv_attn(memory).split(self.n_emb, dim=-1)
        Q = self.q_attn(x)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        K = K.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        V = V.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        # Pass ELU to the Q, K - per Eq. 19 from Sue, et al
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # applying the rotary positional embedding to Q and K
        Q_rot = self.pos_encoding(Q)
        K_rot = self.pos_encoding(K)

        # causal self-attention; cross-attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S)
        denom = Q @ K.transpose(-2, -1)
        att = Q_rot @ K_rot.transpose(-2, -1)
        att = att / denom.sum(dim=-1, keepdim=True)
        att = self.attn_dropout(att)
        y = att @ V # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, nh, hs)

        # output projection
        y = self.res_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_emb, config.n_emb * 4, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh') # do a GELU approximation for faster computation
        self.c_proj = nn.Linear(config.n_emb * 4, config.n_emb, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.self_attn = CausalSelfAttention(config, True)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x, memory):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), memory)
        x = x + self.mlp(self.ln_3(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        assert config.vocab_size is not None
        assert config.max_len is not None
        self.config = config 

        # create the critical components of the transformer
        self.transformer = nn.ModuleDict(dict(
            src_wte = nn.Embedding(config.vocab_size, config.n_emb),
            tgt_wte = nn.Embedding(config.vocab_size, config.n_emb),
            enc = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            dec = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        # the final projection layer
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)
        self.transformer.src_wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.transformer.tgt_wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights_)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params_()/1e6,))

    def get_num_params_(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def _init_weights_(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt):
        _, t = src.size()
        assert t <= self.config.max_len, f"Cannot forward sequence of length {t}, max length is only {self.config.max_len}"

        # forward the encoder block
        src_emb = self.transformer.src_wte(src)
        for block in self.transformer.enc:
            src_emb = block(src_emb)

        # forward the decoder block
        tgt_emb = self.transformer.tgt_wte(tgt)
        for block in self.transformer.dec:
            tgt_emb = block(tgt_emb, src_emb)

        tgt_emb = self.transformer.ln_f(tgt_emb)
        return self.lm_head(tgt_emb)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_len else idx[:, -self.config.max_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
@dataclass
class TransformerConfig:
    n_emb: int
    n_head: int
    n_layer: int
    max_len: int
    vocab_size: int
    dropout: float
    bias: bool 
    
if __name__ == "__main__":
    model_config = TransformerConfig(
        n_emb = 512,
        n_head = 8,
        n_layer = 6,
        max_len = 512,
        vocab_size = 37467,
        dropout = 0.1,
        bias = True

    )
    model = Transformer(model_config)
    a = torch.randint(0, 37467, (32, 12))
    b = torch.randint(0, 37467, (32, 16))
    model.eval()
    output = model(a, b)
    print(output.shape)