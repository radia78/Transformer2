import torch
import torch.nn.functional as F
from model import Transformer, TransformerConfig
from transformers import PreTrainedTokenizerFast

class Generator:
    def __init__(self, ckpt_path, model_config, device):
        # load the model
        self.model = self.load_model(ckpt_path, model_config, device)
        self.model.eval()

        # initialize tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('radia/wmt14-de2en-tokenizer')
        print()

        self.device = device

    def load_model(self, ckpt_path, model_config, device):
        model = Transformer(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        model.to(device)
        return model
    
    def decode(self, src_sent, max_len, temperature, k, decode_method='greedy'):

        # send the src tokens to device
        src_tokens = self.tokenizer.encode(src_sent, return_tensors='pt')

        if decode_method == 'greedy':
            result = self.greedy_search(src_tokens, max_len, temperature)
            result = self.tokenizer.decode(result.squeeze(0))
        
        if decode_method == 'beam':
            result = self.beam_search(src_tokens, max_len, k, temperature)
            result = self.tokenizer.decode(result[0][1].squeeze(0))

        return result
    
    def greedy_search(self, src_tokens, max_len, temperature=1.0):

        # cache the number of batches
        b = src_tokens.shape[0]
        # create a target dummy with only the start of sentence token
        idx = torch.zeros(b, 1, device=self.device).long()
        # send the source tokens into the same device
        src_tokens = src_tokens.to(self.device)

        for _ in range(max_len):

            # forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits = self.model(src_tokens, idx)
                logits = logits[:, -1, 1:] / temperature

                if b == 1:
                    logits = logits.unsqueeze(0)

            # choose the most likely guest
            idx_next = logits.argmax(2)

            # see if the next token is the end of sentence token
            if idx_next.item() == 2:
                break

            else:
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

        return idx.detach()

    def beam_search(self, src_tokens, max_len, k=5, temperature=1.0):
        # cache the number of batches
        b = src_tokens.shape[0]

        # create a list of candidate tensors
        candidates = []

        # create a temporary cache for hypothesis
        hypotheses = [(0, torch.zeros(b, 1, device=self.device).long())]

        for n in range(max_len):
            for i in range(len(hypotheses)):

                # forward the model to get the logits for the index in the sequence
                with torch.no_grad():
                    logits = self.model(src_tokens, hypotheses[i][1])
                    logits = logits[:, -1, 1:] / temperature

                # choose the top 4 candidate
                scores, indices = logits.topk(k, dim=-1, largest=True, sorted=True)
                scores += hypotheses[i][0]

                if b == 1:
                    indices = indices.unsqueeze(0)

                for j in range(k):
                    # append candidates
                    hypotheses.append(
                        (scores[0][j].item(), torch.cat((hypotheses[i][1], indices[:, 0, j].unsqueeze(0)), dim=-1))
                    )
            
            # pop out the initial value
            if n == 0:
                hypotheses = hypotheses[1:]

            # pruning the list
            hypotheses = sorted(hypotheses, reverse=True)
            hypotheses = hypotheses[:4]

            # add the hypotheses list to candidates
            candidates += hypotheses

        # normalize the score
        candidates = list(map(lambda x: (x[0]/x[1].shape[-1]**0.6, x[1]), candidates))
        candidates = sorted(candidates, reverse=True)

        return candidates

if __name__ == "__main__":
    # instatiate the model configuration
    model_config = TransformerConfig(
        n_emb = 512,
        n_head = 8,
        n_layer = 6,
        max_len = 512,
        vocab_size = 31556,
        dropout = 0.1
    )

    # creating the source tokens
    src_sent = "Ich bin trinke Bier."

    generator = Generator('chkpt.pt', model_config, 'cpu')
    result = generator.decode(src_sent, 128, 1.0, 4, 'beam')
    print(result)
