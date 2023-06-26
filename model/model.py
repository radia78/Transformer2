from blocks import *
import torch.functional as F
from dataclasses import dataclass

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