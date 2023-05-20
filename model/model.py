from blocks import *
import torch.functional as F

class RopeBart(nn.module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        self.config = config
        self.n_layer = config.n_layer

        self.transformer = nn.ModuleDict(dict(
            ln_f = nn.LayerNorm(config.n_embd),
            src_emb = nn.Embedding(config.vocab_size, config.n_emb),
            tgt_emb = nn.Embedding(config.vocab_size, config.n_emb),
            enc = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            dec = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        print("number of parameters: ")

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
                y = self.transformer.dec[i](y, x)
            # normalize before final linear layer
            x = self.transformer.ln_f(x)
            return self.lm_head(x)