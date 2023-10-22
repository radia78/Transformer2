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

        self.device = device

    def load_model(self, ckpt_path, model_config, device):
        model = Transformer(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        model.to(device)
        return model
    
    def decode(self, src_sent, max_len, temperature, decode_method='greedy'):

        # send the src tokens to device
        src_tokens = self.tokenizer.encode(src_sent, return_tensors='pt')

        if decode_method == 'greedy':
            result = self.greedy_search(src_tokens, max_len, temperature)

        result = self.tokenizer.decode(result.squeeze(0))

        return result
    
    def greedy_search(self, src_tokens, max_len, temperature=1.0):

        # create a target dummy with only the start of sentence token
        idx = torch.zeros(1, 1, device=self.device).long()
        # send the source tokens into the same device
        src_tokens = src_tokens.to(self.device)

        for _ in range(max_len):

            # forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits = self.model(src_tokens, idx)
                logits = logits [:, -1, :] / temperature

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # see if the next token is the end of sentence token
            if idx_next.item() == 2:
                break

            else:
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

        return idx.detach()

if __name__ == "__main__":
    # instatiate the model configuration
    model_config = TransformerConfig(
        n_emb = 512,
        n_head = 8,
        n_layer = 6,
        max_len = 512,
        vocab_size = 37467,
        dropout = 0.1
    )

    # creating the source tokens
    src_sent = "Ich bin trinke Bier."

    generator = Generator('TransformerV2/chkpt.pt', model_config, 'mps')
    result = generator.decode(src_sent, 512, 1.0)
    print(result)
