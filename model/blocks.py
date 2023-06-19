import math
from attention import *

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
    