import torch
from torch import nn
import math

'''
Author: Radi Akbar
Date: 4/11/2023
'''

class MiniBartEUW(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 maxlen:int,
                 dmodel:int,
                 ff_size:int,
                 nlayer:int,
                 nhead:int,
                 dropout:float,
                 pad_idx:int,
                 device
                 ):
        super(MiniBartEUW, self).__init__()

        self.device = device
        self.pad_idx = pad_idx

        # Embedding layers
        self.emb_pos = PositionalEncoding(dmodel, dropout, maxlen)
        self.src_embedding = nn.Embedding(vocab_size, dmodel)
        self.tgt_embedding = nn.Embedding(vocab_size, dmodel)

        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=dmodel, 
            nhead=nhead, 
            num_encoder_layers=nlayer,
            num_decoder_layers=nlayer,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            norm_first=True
        )
        self.final_linear = nn.Linear(dmodel, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0., std=0.02)

    def make_padding_masks(self, src, tgt):
        src_padding_mask = (src == self.pad_idx).transpose(0, 1) # (N, src_len)
        src_padding_mask = torch.where(src_padding_mask, torch.tensor(float('-inf')), torch.tensor(0.))

        tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1) # (N, tgt_len)
        tgt_padding_mask = torch.where(tgt_padding_mask, torch.tensor(float('-inf')), torch.tensor(0.))

        return src_padding_mask.to(self.device), tgt_padding_mask.to(self.device)
    
    def forward(self, src, tgt):

        # Create the masks
        src_padding_mask, tgt_padding_mask = self.make_padding_masks(src, tgt)
        peaking_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0], self.device)
        
        # Create token embeddings
        src_emb = self.emb_pos(self.src_embedding(src))
        tgt_emb = self.emb_pos(self.tgt_embedding(tgt))
        
        # Transformer outputs
        decoder_outputs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=peaking_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
            )

        # Transformer block
        outputs = self.final_linear(decoder_outputs)
        
        return outputs
 
class PositionalEncoding(nn.Module):
    '''
    Args:
        maxlen: Maximum length of the sequence
        dmodel: Dimension of the embedding
        dropout: Dropout rate after adding the word embedding and position embedding
    '''
    def __init__(self, dmodel:int, dropout: float, maxlen:int=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, dmodel, 2)* math.log(10000) / dmodel)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, dmodel))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding:torch.Tensor):
        '''
        Args:
            token_embedding: The embedded tensor of tokens (Seq Len, Batch Size, Embedding Size)
            
        Returns:
            x: The word embedding + positional encoding and applied dropout (Seq Len, Batch Size, Embedding Size)
        '''
        x = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return x