from datasets import load_dataset
import numpy as np
import random
from itertools import chain
from functools import reduce
import re
from nltk.tokenize import sent_tokenize
from transformers import MBart50TokenizerFast

import torch

""" dataset = load_dataset("wikipedia", "20220301.simple")
tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang="en_XX") """

""" def _text_to_tokens(text):
    return list(
        map(
            lambda x: tokenizer(x)['input_ids'][1:],
            list(
                 map(
                     lambda x: sent_tokenize(x),
                     text.split('\n')
                 )
            )
        )
    )

def _check_token_len(sentences):
    return sum(
        list(
            map(
                lambda x: len(x),
                sentences
            )
        )
    )

def _corrupt_text(sentences):
    span_len = np.random.poisson(lam=3.0)
    selected_spans = random.sample(sentences, min(span_len, len(sentences)))
    return reduce(
        lambda x,y : x + y, 
        [[tokenizer.mask_token_id] if x in selected_spans else x for x in sentences]
    )

def process(example):
    # convert the text into a nested list of tokens for each sentence
    sentences = _text_to_tokens(example['text'])
    num_sentences = len(sentences)
    tot_token_len = _check_token_len(sentences)
    if tot_token_len >= 512 and num_sentences == 1:

        tgt = []
        token_lens = 0
        
        # take only sentences that add-up to 512 tokens
        for i, sentence in enumerate(sentences):
            token_lens += len(sentence)
            if token_lens < 512:
                tgt += sentence
            else:
                break

        return {"tgt": tgt, "len": len(sentences[:i])}
    
    else:
        return {"tgt": list(chain.from_iterable(sentences)), "len": len(sentences)} """
    
""" tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=6,
) """
""" sample_text = dataset['train']['text'][1085]
new_dataset = map(
        lambda x: sent_tokenize(x.lstrip(' ').rstrip(' ')),
        filter(
            lambda x: True if x != '' else False,
            sample_text.split('\n')
    )
)
new_new_dataset = map(
    lambda x: tokenizer(x)['input_ids'],
    new_dataset,
)
print(list(new_new_dataset)) """