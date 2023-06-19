import torch
import random
from torch.nn.utils.rnn import pad_sequence

"""
Create a class that given a certain number of pretraining tasks, create a collating function that incorporates those tasks in the same class
"""

class MultiTaskCollator:
    def __init__(self, tokenizer, mask_rate: float=0.3, poisson_lambda: float=3.0):
        self.tokenizer = tokenizer
        # make sure the arguments make sense
        assert mask_rate >= 0.0 and mask_rate <=1.0
        assert poisson_lambda >=0.0
        self.mask_rate = mask_rate
        self.poisson_lambda = poisson_lambda

    # documentation-level tasks
    def DocumentLevelTask(self, batch):
        input_batch, target_batch = [], []
        for targets in batch:
            inputs_text = targets['input_ids']

            span_length = int(torch.distributions.uniform.Uniform(0.5 * batch['len'], batch['len']).item())
            start_index = int(torch.distributions.uniform.Uniform(0, batch['len'] - span_length).item())
            inputs_text[start_index:start_index + span_length] = self.tokenizer.mask_token_id

            input_batch.append(inputs_text)
            target_batch.append(targets['input_ids'])

        input_batch = pad_sequence(input_batch, padding_value=self.tokenizer.pad_token_id)
        target_batch = pad_sequence(target_batch, padding_value=self.tokenizer.pad_token_id)
        return input_batch, target_batch
    
    def TextInfillingTask(self, batch):
        input_batch, target_batch = [], []
        for targets in batch:
            mask_tokens = []
            inputs_text = targets['input_ids']

            mask_budget = targets['len']
            while (mask_budget > int(self.mask_rate * targets['len'])):
                span_length = int(torch.distributions.poisson.Poisson(3.0).item())
                if span_length > mask_budget:
                    span_length = targets['len']

                start_index = int(torch.distributions.uniform.Uniform(0, mask_budget - span_length).item())
                mask_tokens.append(self.tokenizer.mask_token_id)
                inputs_text = inputs_text[:start_index] + inputs_text[start_index + span_length:]
                mask_budget =- span_length

            inputs_text += mask_tokens
            random.shuffle(inputs_text)

            input_batch.append(inputs_text)
            target_batch.append(targets)

        input_batch = pad_sequence(input_batch, padding_value=self.tokenizer.pad_token_id)
        target_batch = pad_sequence(target_batch, padding_value=self.tokenizer.pad_token_id)
        return input_batch, target_batch
