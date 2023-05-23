from transformers import BartTokenizerFast 
from datasets import load_dataset 

dataset = load_dataset('wikipedia', '20220301.simple')
enc = BartTokenizerFast.from_pretrained('facebook/bart-base')

sample_text = dataset['train']['text'][0]
sample_tokens = enc.encode(sample_text, truncation=True, max_length=512)