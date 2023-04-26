import pickle
import torch
import os
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MiniBartPreTraining(Dataset):
    def __init__(self, 
                 dataset_path=None,
                 dataset_filename=None,
                 spm_model_path=None,
                 spm_vocab_path=None,
                 languages=['en', 'fr', 'de']
                 ):
        
        # Checking if the tokenizer path has expired
        try:
            self.tokenizer = T.SentencePieceTokenizer(r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model")
            self.vocab = T.VocabTransform(load_state_dict_from_url(r"https://download.pytorch.org/models/text/xlmr.vocab.pt")) 
        except:
            try:
                self.tokenizer = T.SentencePieceTokenizer(spm_model_path)
                self.vocab = T.VocabTransform(spm_vocab_path)
            except:
                print("Invalid model or vocab path.")

        for l in languages:
            self.vocab.vocab.append_token(f'<{l}>')

        full_filepath = os.path.join(dataset_path, dataset_filename)
        with open(full_filepath, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return (len(self.data['target']))
    
    def __getitem__(self, index):
        
        source_tokens = []
        target_tokens = []

        for sentence in self.data['source'][index][:-1]:
            source_tokens += list(map(lambda x: x.replace('▁_', '<mask>'), self.tokenizer(sentence))) + ['</s>']
        source_tokens += [self.data['source'][index][-1]]

        for sentence in self.data['target'][index][1:]:
            target_tokens += list(map(lambda x: x.replace('▁_', '<mask>'), self.tokenizer(sentence))) + ['</s>']
        target_tokens = [self.data['target'][index][0]] + target_tokens

        return self.vocab(source_tokens), self.vocab(target_tokens)
    
def text_transforms(tensor, maxlen):
    transforms = T.Sequential(
        T.Truncate(maxlen),
        T.ToTensor() 
        )
    return transforms(tensor)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])