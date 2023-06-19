from tokenizers import Tokenizer
from tokenizers.models import BPE
from typing import List
from dataclasses import dataclass, field
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

class MachineTranslationTokenizer:
    def __init__(self, config):
        self.unk_token = config.unk_token
        self.spl_tokens = config.spl_tokens

    def prepare_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token)) # declare a bpe tokenizer
        trainer = BpeTrainer(
            special_tokens=self.spl_tokens,
            end_of_word_suffix="_",
            vocab_size=37461 # the number of vocab for consistence with "Attention is All You Need"
            ) # create a trainer for the bpe tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer, trainer

    def train_tokenizer(self, files:List[str], save_dir:str):
        tokenizer, trainer = self.prepare_tokenizer()
        tokenizer.train(files, trainer)
        tokenizer.save(f"{save_dir}.json")

    @staticmethod
    def load_tokenizer(save_dir, max_len=512):
        tokenizer = Tokenizer.from_file(f"{save_dir}.json")
        tokenizer.decoder = BPEDecoder(suffix="_")
        tokenizer.enable_truncation(max_length=512)
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))]
        )
        return tokenizer

@dataclass
class TokenizerConfig:
    unk_token:str = "<unk>" # token for unknown words
    spl_tokens:list[str] = field(default_factory=list)

if __name__ == "__main__":
    file_names = [
        "data/mt/wmt14_training/train.de",
        "data/mt/wmt14_training/train.en"
    ]
    tok_config = TokenizerConfig(spl_tokens=["<s>", "<pad>", "</s>", "<mask>", "<unk>"])
    tokenizer = MachineTranslationTokenizer(tok_config) # initiate class
    tokenizer.train_tokenizer("data/mt/transformer-mt-base") # train the tokenizer
