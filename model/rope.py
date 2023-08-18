import torch
import torch.nn as nn

"""
Implementation of the rotary positional embedding
lifted directly from https://blog.eleuther.ai/rotary-embeddings/
"""

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
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
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :] # (batch_size, seq_len, nheads, nemb)
            self.sin_cached = emb.sin()[None, :, None, :] # (batch_size, seq_len, nheads, nemb)
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

if __name__ == "__main__":
    a = torch.randn(3, 2, 3, 4) # batch, seq_len, num_heads, emb_dim
    print(f"The embedding with shapes {a.shape}: \n", a)

    # rotate half of the embedding space
    rot_a = rotate_half(a)
    print(f"The rotated embedding with shapes {rot_a.shape}: \n", rot_a)

    # create the cos and sin
    cos, sin = Rotary(a.shape[-1])(a)
    pos_a = apply_rotary_pos_emb(rot_a, cos, sin)

    assert a.shape == pos_a.shape

    # print the embedding
    print(f"The final position imbued embeddings: \n", pos_a)
