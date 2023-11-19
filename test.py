from utils import Generator
from model import TransformerConfig
import time

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
    src_sents = "Ich erkläre die am Freitag, dem 17. Dezember unterbrochene Sitzungsperiode des Europäischen Parlaments für wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum Jahreswechsel und hoffe, daß Sie schöne Ferien hatten."
    tgt_sents = "I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period."
    
    generator = Generator('models/TransformerV2/chkpt.pt', model_config, 'cpu')
    src_ids = generator.tokenizer(src_sents, return_tensors='pt')
    tgt_ids = generator.tokenizer(tgt_sents, return_tensors='pt')
    print(src_ids.input_ids.shape)
    print(tgt_ids.input_ids.shape)
    t0 = time.time()
    results = generator.decode(src_sents, max_len=512, temperature=1.0, k=5, decode_method='greedy')
    print(results)
    t1 = time.time()
    print(t1 - t0)