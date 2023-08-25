import torch
import math
import logging
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
RMS Norm implementation from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
ROPE implementation from https://blog.eleuther.ai/rotary-embeddings/
"""

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, n, device):
        seq_len = n
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(n, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

# Biao Zhang's implementation of RMS Norm
class RMSNorm(nn.Module):
    def __init__(self, d, p=0.00625, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    
# implementation of SwiGLU from lucidrains
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# implement the self attention mechanism for rotated embeddings
class MultiHeadAttention(nn.Module):
    def __init__(self, config, causal=False):
        super(MultiHeadAttention, self).__init__()
        # cache the dimension of the fused parallel attention
        self.n_emb = config.n_emb
        self.n_head = config.n_head

        # create the projection layers
        self.fused_kv_proj = nn.Linear(config.n_emb, 2 * config.n_emb, bias=False)
        self.q_proj = nn.Linear(config.n_emb, config.n_emb, bias=False)
        self.attn_out = nn.Linear(config.n_emb, config.n_emb) # output projection

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid = nn.Dropout(config.dropout)

        # save the booleans
        self.causal=causal
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # create the mask if flash attention ain't available
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.max_len, config.max_len))
                                        .view(1, 1, config.max_len, config.max_len))
            
    def pos_encoding(self, x):
        cos, sin = Rotary(self.n_emb // self.n_head)(x.shape[2], x.device)
        return apply_rotary_pos_emb(x, cos, sin)

    def forward(self, x, m=None):
        """
        b - batch
        h - heads
        n, i, j - sequence_lengths
        d - feature dimension
        """
        n, h = x.shape[1], self.n_head
        if m is None: # just in case there's cross - attention
            m = x

        # calculate q, k, v
        k, v = self.fused_kv_proj(m).split(self.n_emb, dim=-1)
        q = self.q_proj(x)
        q = rearrange(q, "b i (h d) -> b h i d", h=h)
        k = rearrange(k, "b j (h d) -> b h j d", h=h)
        v = rearrange(v, "b j (h d) -> b h j d", h=h)

        # applying the rotary positional embedding to Q and K
        q = self.pos_encoding(q)
        k = self.pos_encoding(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S)
        if self.flash: # flash attention go brrrrrrrrrr
            # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
            y = rearrange(y, "b h i d -> b i (h d)")
        else:
            att = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                 att = att.masked_fill(self.bias[:,:,:n,:n] == 0, float('-inf'))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = torch.einsum("b h i j, b h j d, -> b h i d", att, v) # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
            y = rearrange(y, "b h i d -> b i (h d)")

        return self.attn_out(y)
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        ff_inner_dim = int(math.ceil(config.n_emb * 2/3 * 4)) # following SwiGLU recommendation of 2/3 * 4 * d
        self.proj_1 = nn.Linear(config.n_emb, 2 * ff_inner_dim, bias=False)
        self.swiglu = SwiGLU()
        self.proj_2 = nn.Linear(ff_inner_dim, config.n_emb, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.swiglu(x)
        x = self.proj_2(x)

        return self.dropout(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.ln_1 = RMSNorm(config.n_emb)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = RMSNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.ln_1 = RMSNorm(config.n_emb)
        self.attn1 = MultiHeadAttention(config, causal=True)
        self.ln_2 = RMSNorm(config.n_emb)
        self.attn2 = MultiHeadAttention(config)
        self.ln_3 = RMSNorm(config.n_emb)
        self.mlp = MLP(config)
    
    def forward(self, x, m):
        x = x + self.attn1(self.ln_1(x))
        x = x + self.attn2(self.ln_2(x), m)
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
        self.apply(self._init_weights)

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
    
@dataclass
class TransformerConfig:
    n_emb: int
    n_head: int
    n_layer: int
    max_len: int
    vocab_size: int
    dropout: float
    
if __name__ == "__main__":
    model_config = TransformerConfig(
        n_emb = 512,
        n_head = 8,
        n_layer = 6,
        max_len = 512,
        vocab_size = 37467,
        dropout = 0.1
    )
    model = Transformer(model_config)
    a = torch.randint(0, 37467, (32, 12))
    b = torch.randint(0, 37467, (32, 16))
    model.eval()
    output = model(a, b)
    print(output.shape)