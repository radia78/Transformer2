from rope import *
import torch.nn as nn

class ROPESelfAttention(nn.Module):
    def __init__(self, config):
        super(ROPESelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # positional embedding layer
        self.rotary_emb = Rotary(config.n_emb)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, causal:bool):
        B, T, C = x.size() # T, batch_size, embedding_dimensionality

        # calculate query, key, values for all heads in batch
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # inject rotary positional embedding
        cos, sin = self.rotary_emb(q, seq_dim=2)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        v = apply_rotary_pos_emb(v, cos, sin)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class ROPECrossAttention(nn.MOdule):
    def __init__(self, config):
        super(ROPECrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # positional embedding layer
        self.rotary_emb = Rotary(config.n_emb)
        # query projection for each head in a batch
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # key, value projection for each head in a batch
        self.kv_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, m):
        B, T, C = x.size() # source sequence len, batch_size, embedding_dimensionality
        _, S, _ = m.size() # target sequence len

        # calculate query, key, values for all heads in batch
        q = self.q_attn(x)
        k, v = self.kv_attn(m).split(2, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        # inject rotary positional embedding
        cosq, sinq = self.rotary_emb(q, seq_dim=2)
        coskv, sinkv = self.rotary_emb(k, seq_dim=2)
        q = apply_rotary_pos_emb(q, cosq, sinq)
        k = apply_rotary_pos_emb(k, coskv, sinkv)
        v = apply_rotary_pos_emb(v, coskv, sinkv)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    