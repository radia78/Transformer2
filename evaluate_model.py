import datasets
from model import TransformerConfig
from utils import Generator
from datasets import load_dataset

def predict(generator, dataset, langs, max_len, temp, k, decode_method):
    # initalize the list of translations
    predictions = []
    references = []

    # loop through the observations
    for sent in dataset:
        # append the prediction
        predictions.append(
            generator.decode(sent[langs[0]], max_len, temperature=temp, k=k, decode_method=decode_method)
        )
        # append the actual translation
        references.append([sent[langs[1]]])

    return predictions, references

def main(args):

    # load the sacrebleu metric
    ds = load_dataset("wmt14", f"{args.langs[0]}-{args.langs[1]}", split="test")['translation']

    # create the model config
    model_config = TransformerConfig(
        args.n_emb,
        args.n_head,
        args.n_layer,
        args.max_len,
        args.vocab_size,
        args.dropout
    )

    # load the metric to score bleu
    metric = datasets.load_metric("sacrebleu")

    # create the generator class
    generator = Generator(args.ckpt_path, model_config, args.device)

    # get the list of references and predictions
    predictions, references = predict(generator, ds, args.langs, args.max_len, args.temp, args.k, args.decode_method)

    # compute the score
    results = metric.compute(predictions=predictions, references=references)

    print(results)

if __name__ == "__main__":
    import argparse

    # setup the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # model config arguments
    args.n_emb = 512
    args.n_head = 8
    args.n_layer = 6
    args.max_len = 512
    args.vocab_size = 37467
    args.dropout = 0.1
    
    # setup configurations
    args.langs = ['de', 'en']
    args.ckpt_path = "models/TransformerV2/chkpt.pt"
    args.device = "cpu"
    args.temp = 1.0
    args.k = 5 # following the original paper, we use beam size of 5
    args.decode_method = 'greedy'

    # run the test
    main(args)