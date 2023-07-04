import torch
import math
import torch.nn as nn
from dataclasses import dataclass
import unittest
import numpy as np
from einops import repeat

# rotary positional embedding class (generate cos & sin)
class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super(Rotary, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = repeat(freqs, "b n -> b (n j)", j=2).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:
def rotate_every_two(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).view(x.shape)

@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_every_two(x) * sin)

# define a unittest for the positional embedding layer
class ROPEUnitTest(unittest.TestCase):
    def test_shapes(self):
        # apply positional embedding
        test_input = torch.randn(16, 4, 64, 64)
        rope = Rotary(test_input.shape[-1])
        cos, sin = rope(test_input, seq_dim=2)
        output = apply_rotary_pos_emb(test_input, cos, sin)

        self.assertTrue(output.shape == test_input.shape, "Input shape and output shape must match")

    def test_rotation(self):
        # define a helper function
        def rotate_matrix(x, m:int):
            """Rotate a vector by theta 1 counter clockwise (theta 1 is the angle of the first 2 embeddings)"""
            assert x.shape == torch.Size([1, 2]), "'x' is not a 2-length vector"

            # compute components
            theta = 10000**(-2*(1 - 1)/2)
            sin_comp = np.sin(m * theta)
            cos_comp = np.cos(m * theta)
            rotation_matrix = torch.tensor([[cos_comp, sin_comp], [-sin_comp, cos_comp]]).float()

            return torch.matmul(x, rotation_matrix)
        
        # define the test case 1: [1, 0], [1, 0]
        test_input1 = torch.tensor([[1, 0], [0, 1]]).float()

        # compute using the positional embedding
        rope1 = Rotary(test_input1.shape[-1])
        cos1, sin1 = rope1(test_input1.unsqueeze(0).unsqueeze(1), seq_dim=2)
        module_output1 = apply_rotary_pos_emb(test_input1, cos1, sin1)

        # compute using rotational matrix
        expected_output1 = torch.cat(
            [rotate_matrix(test_input1[i].reshape(1, 2), i) for i in range(test_input1.shape[0])]
        )

        # test case 1
        self.assertTrue(torch.allclose(module_output1, expected_output1)), "Test 1 Failed: Vector is not rotated by theta1"

        # define the test case 2: random input of length 10
        test_input2 = torch.randn(10, 2)

        # compute using positional embedding
        rope2 = Rotary(test_input2.shape[-1])
        cos2, sin2 = rope2(test_input2.unsqueeze(0).unsqueeze(1), seq_dim=2)
        module_output2 = apply_rotary_pos_emb(test_input2, cos2, sin2)

        # compute using rotational matrix
        expected_output2 = torch.cat(
            [rotate_matrix(test_input2[i].reshape(1, 2), i) for i in range(test_input2.shape[0])]
        )

        # test case 2
        self.assertTrue(torch.allclose(module_output2, expected_output2)), "Test 2 Failed: Vector is not rotated by theta1"     

# self attention layer
class ROPESelfAttention(nn.Module):
    def __init__(self, config):
        super(ROPESelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension is not divisble by number of heads"
        # positional embedding layer
        self.rotary_emb = Rotary(config.n_embd // config.n_head)
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
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs), doesn't get rotated

        # inject rotary positional embedding
        cos, sin = self.rotary_emb(q, seq_dim=2)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# cross attention layer
class ROPECrossAttention(nn.Module):
    def __init__(self, config):
        super(ROPECrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # positional embedding layer
        self.rotary_emb = Rotary(config.n_embd // config.n_head)
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
        k, v = self.kv_attn(m).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        # inject rotary positional embedding
        cosq, sinq = self.rotary_emb(q, seq_dim=2)
        cosk, sink = self.rotary_emb(k, seq_dim=2)
        q = apply_rotary_pos_emb(q, cosq, sinq)
        k = apply_rotary_pos_emb(k, cosk, sink)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# define the dataclass for testing
@dataclass
class TestConfig:
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

# define unit testing for attention
class AttentionUnitTest(unittest.TestCase):
    def test_self_attention_shape(self):
        test_input = torch.randn(32, 16, 512) # batch size, seq len, emb size
        config = TestConfig()
        self_attention = ROPESelfAttention(config)
        output = self_attention(test_input, False)

        self.assertTrue(output.shape == test_input.shape)

    def test_cross_attention_shape(self):
        test_input = torch.randn(32, 16, 512) # test the target sequence
        test_memory_input = torch.randn(32, 14, 512) # test the memory sequence
        config = TestConfig() # create the test config
        
        cross_attention = ROPECrossAttention(config)
        output = cross_attention(test_input, test_memory_input)

        self.assertTrue(output.shape == test_input.shape)

@torch.jit.script
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = ROPESelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), False)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.self_attn = ROPESelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = ROPECrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, m):
        x = x + self.self_attn(self.ln_1(x), True)
        x = x + self.cross_attn(self.ln_2(x), m)
        x = x + self.mlp(self.ln_3(x))
        return x
    
class TransformerBlockUnitTest(unittest.TestCase):
    def test_encoder_block_shape(self):
        test_input = torch.randn(32, 16, 512) # random input
        config = TestConfig() # initiate config
        encoder = EncoderBlock(config) # encoder
        output = encoder(test_input) # output of the encoder

        self.assertTrue(output.shape == test_input.shape)

    def test_decoder_block_shape(self):
        test_input = torch.randn(32, 16, 512) # test input
        test_memory_input = torch.randn(32, 14, 512) # test memory input
        config = TestConfig() # initiate config
        decoder = DecoderBlock(config) # initiate the decoder
        output = decoder(test_input, test_memory_input) # output

        self.assertTrue(output.shape == test_input.shape)

# ropeformer model
class RopeFormer(nn.Module):
    def __init__(self, config):
        super(RopeFormer, self).__init__()
        # assert the vocabulary size
        assert config.vocab_size is not None
        self.config = config
        self.n_layer = config.n_layer
        # assemble the transformer network
        self.transformer = nn.ModuleDict(dict(
            ln_f = nn.LayerNorm(config.n_embd),
            src_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_idx),
            tgt_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_idx),
            enc = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            dec = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        print(f"number of parameters: {self.get_num_params(False)}")

        # init all weights
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= (self.transformer.src_emb.weight.numel() + self.transformer.tgt_emb.weight.numel())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt):
        # pass the source and target to the embeddings
        x, y = self.transformer.src_emb(src), self.transformer.tgt_emb(tgt)
        for i in range(self.n_layer):
            x = self.transformer.enc[i](x)
        for i in range(self.n_layer):
            y = self.transformer.dec[i](y, x)
        # normalize before final linear layer
        y = self.transformer.ln_f(y)
        return self.lm_head(y)
        
@dataclass
class RopeBARTConfig:
    pad_idx: int = 1
    vocab_size: int = 37467 # number of vocab corresponding to 32k merges
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

# define unit test for the model
class RopeFormerTest(unittest.TestCase):
    def test_rope_model_shape(self):
        config = RopeBARTConfig() # initiate config
        model = RopeFormer(config) # initiate model

        # create test inputs
        src = torch.randint(0, 37467, (32, 14))
        tgt = torch.randint(0, 37467, (32, 16))

        # create the model output
        output = model(src, tgt)

        self.assertTrue(output.shape == torch.Size([32, 16, 37467]))

if __name__ == "__main__":
    unittest.main()
