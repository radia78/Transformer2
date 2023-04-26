import random
import pickle
import math
from numpy.random import poisson
from nltk.tokenize import sent_tokenize
from torchtext.datasets import IWSLT2016
import revtok

sentences = {}
sentences['source'] = []
sentences['target'] = []

fr_train_data = IWSLT2016(split='train', language_pair=('fr', 'en'))
de_train_data = IWSLT2016(split='train', language_pair=('de', 'en'))

for fr, en in fr_train_data:
    fr = fr.rstrip('\n')
    sent_seg = sent_tokenize(fr, 'french')
    sentences['target'].append(["<fr>"] + sent_seg)
    tokens = []
    for sentence in sent_seg:
        sent_tok = revtok.tokenize(sentence)
        orig_len = len(sent_tok)
        mask_budget = math.ceil(0.35*orig_len)
        while mask_budget > 0:
            start_pos = random.randint(0, orig_len)
            span = poisson(3.5)
            if (mask_budget - span) < 0:
                end_pos = mask_budget + start_pos
            elif (span + start_pos) > orig_len:
                end_pos = orig_len
            else:
                end_pos = start_pos + span
            sent_tok[start_pos:end_pos] = [" _ "]
            mask_budget -= (end_pos - start_pos)
        tokens.append(revtok.detokenize(sent_tok))
    sentences['source'].append(tokens + ["<fr>"])

    en = en.rstrip('\n')
    sent_seg = sent_tokenize(en, 'english')
    sentences['target'].append(["<en>"] + sent_seg)
    tokens = []
    for sentence in sent_seg:
        sent_tok = revtok.tokenize(sentence)
        orig_len = len(sent_tok)
        mask_budget = math.ceil(0.35*orig_len)
        while mask_budget > 0:
            start_pos = random.randint(0, orig_len)
            span = poisson(3.5)
            if (mask_budget - span) < 0:
                end_pos = mask_budget + start_pos
            elif (span + start_pos) > orig_len:
                end_pos = orig_len
            else:
                end_pos = start_pos + span
            sent_tok[start_pos:end_pos] = [" _ "]
            mask_budget -= (end_pos - start_pos)
        tokens.append(revtok.detokenize(sent_tok))
    sentences['source'].append(tokens + ['<en>'])

for de, en in de_train_data:
    fr = fr.rstrip('\n')
    sent_seg = sent_tokenize(de, 'deutsch')
    sentences['target'].append(['<de>'] + sent_seg)
    tokens = []
    for sentence in sent_seg:
        sent_tok = revtok.tokenize(sentence)
        orig_len = len(sent_tok)
        mask_budget = math.ceil(0.35*orig_len)
        while mask_budget > 0:
            start_pos = random.randint(0, orig_len)
            span = poisson(3.5)
            if (mask_budget - span) < 0:
                end_pos = mask_budget + start_pos
            elif (span + start_pos) > orig_len:
                end_pos = orig_len
            else:
                end_pos = start_pos + span
            sent_tok[start_pos:end_pos] = [" _ "]
            mask_budget -= (end_pos - start_pos)
        tokens.append(revtok.detokenize(sent_tok))
    sentences['source'].append(tokens + ['<de>'])

with open('pretraining_datasets.pkl', 'wb') as file:
    pickle.dump(sentences, file)