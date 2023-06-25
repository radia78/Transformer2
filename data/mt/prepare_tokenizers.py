from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder
from dataclasses import dataclass, field
from typing import List
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import huggingface_hub
from huggingface_hub import HfApi

class MachineTranslationTokenizer:
    def __init__(self, config):
        self.unk_token = config.unk_token
        self.spl_tokens = config.spl_tokens

    def prepare_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token)) # declare a bpe tokenizer
        trainer = BpeTrainer(
            special_tokens=self.spl_tokens,
            end_of_word_suffix="_",
            vocab_size=37467 # the number of vocab for consistence with "Attention is All You Need"
            ) # create a trainer for the bpe tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer, trainer

    def train_tokenizer(self, dataset, save_dir:str, langs:List[str]):
        tokenizer, trainer = self.prepare_tokenizer()
        tokenizer.train_from_iterator([dataset['train'][lang] for lang in langs], trainer)
        tokenizer.decoder = BPEDecoder(suffix="_")
        tokenizer.enable_truncation(max_length=512)
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))]
        )
        tokenizer.save(f"{save_dir}.json")


@dataclass
class TokenizerConfig:
    unk_token:str = "<unk>" # token for unknown words
    spl_tokens:list[str] = field(default_factory=list)

if __name__ == "__main__":
    train_tokenizer = True # train the tokenizer and push it to huggingface

    if train_tokenizer:
        langs = ['de', 'en']
        dataset = load_dataset("radia/wmt14-de2en")
        tok_config = TokenizerConfig(spl_tokens=["<s>", "<pad>", "</s>", "<mask>", "<unk>"])
        tokenizer = MachineTranslationTokenizer(tok_config) # initiate class
        tokenizer.train_tokenizer(dataset, "data/mt/transformer-mt-base", langs) # train the tokenizer

    # logging into huggingfacehub using the access tokens
    access_tokens = "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"
    huggingface_hub.login(access_tokens)
    user_id = HfApi().whoami()["name"]
    tokenizer_config = {
        "model_type": "tokenizer",
        "tokenizer_class": "tokenizers.Tokenizer",
        "pretrained_model_name_or_path": f"{user_id}/wmt14-de2en-tokenizer"
    } # declare the tokenzier config

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="/Users/radia/Projects/ml_projects/MiniBart/data/mt/transformer-mt-base.json")
    tokenizer.push_to_hub(f"{user_id}/wmt14-de2en-tokenizer", model_tag="v1.0", tokenizer_config=tokenizer_config)

