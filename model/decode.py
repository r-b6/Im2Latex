import torch
import numpy as np
import pandas as pd
from beam_search import BeamSearch


class LatexProducer:
    """
    Used for generating sequences for inference, uses beam search to find most likely sequence.

    end token and start token are passed to ensure the decoder doesn't include them
    in the final sequence.
    max_len is the maximum length of sequences generated (by token count)
    and any sequence exceeding this length is truncated at max_len tokens.
    id2token and token2id convert the output from the model's token vocabulary to strings
    """
    def __init__(self, model, vocab, beam_size=5, max_len=150, use_cuda=False):
        self.model = model
        self.id2token = vocab
        self.token2id = {v: k for k, v in vocab.items()}
        self.end_token = 0
        self.start_token = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.beam_size = beam_size
        self.max_len = max_len
        self.searcher = BeamSearch(self.end_token, max_len, beam_size)

    def __call__(self, image_tensors):
        return self.batch_beam_search(image_tensors)

    def batch_beam_search(self, image_tensors):
        """
        :param image_tensors: [B, 1, H, W] tensor containing a batch of grayscale images
        :return: [B, L] tensor containing most likely sequences,
        as well as a [B, 5] tensor containing log probs for 5 most likely sequences. Log probs of
        most likely sequences is [:, 0] of the log_probs tensor
        """
        image_tensors = image_tensors.to(self.device)
        flat_conv_output = self.model.encode(image_tensors)
        h_0, c_0, o_0 = self.model.get_init_states(flat_conv_output)
        start_predictions = torch.ones(image_tensors.size(0), device=self.device).long() * self.start_token
        state = {'h': h_0, 'c': c_0, 'o': o_0, 'conv': flat_conv_output}
        top_k_predictions, log_probs = self.searcher.search(
            start_predictions, state, self.take_step)
        top_predictions = top_k_predictions[:, 0, :]
        return top_predictions, log_probs

    def take_step(self, last_predictions, curr_state):
        """
        step function passed to beam searcher, takes in the last tokens and a dictionary of LSTM
        states for the batch and returns a [V] tensor containing the log probabilities of the next token and
        a dictionary containing new states for the LSTM
        :param last_predictions: [B]
        :param curr_state: dictionary containing o, h, c hidden states for LSTM as well as the
        output of the conv layer
        :return:
        """
        o_t = curr_state['o']
        h_t = curr_state['h']
        c_t = curr_state['c']
        flat_conv_output = curr_state['conv']
        #running through LSTM for one timestep
        h_new, c_new, o_new, distribution = self.model.decode_t(h_t, c_t, o_t, last_predictions, flat_conv_output)
        curr_state['o'] = o_new
        curr_state['h'] = h_new
        curr_state['c'] = c_new
        #getting log probs
        distribution = torch.log(distribution)
        return (distribution, curr_state)

    def get_formulas(self, top_predictions):
        """
        takes in top predictions are returns formulas as a list of strings with length B
        :param top_predictions: [B, L] tensor
        :return: list containing decoded sequences
        """
        formulas = []
        #iterate through batches
        for curr_sequence in top_predictions:
            curr_sequence = curr_sequence.tolist()
            output = []
            #iterate through tokens in one batch
            for id_ in curr_sequence:
                if id_ == self.start_token:
                    continue
                elif id_ == self.end_token:
                    break
                else:
                    output.append(self.id2token[id_])
            formulas.append(" ".join(output))
        return formulas
