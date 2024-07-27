import torch
import numpy as np
import pandas as pd
from score import score
import os
from decode import LatexProducer

class Tester():
    def __init__(self, model, test_loader, vocab, args):
        self.model = model
        self.test_loader = test_loader
        self.args = args
        self.producer = LatexProducer(self.model, vocab, use_cuda=args.cuda)
        self.vocab = vocab
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            total_bleu = 0
            total_matches = 0
            total_dist = 0
            total_len = 0
            total_samples = 0
            total_batches = 0
            for loader in self.test_loader:
                for images, formulas in loader:
                    references = formulas[:, 1:]
                    images = images.to(self.device)
                    formulas = formulas.to(self.device)
                    references = references.to(self.device)
                    predictions, _ = self.producer(images)
                    total_samples += formulas.shape[0]
                    total_batches += 1
                    bleu, match, dist, length = score(predictions, references, 0)
                    total_bleu += bleu
                    total_matches += match
                    total_dist += dist
                    total_len += length

            bleu_score = total_bleu/total_batches
            exact_match = total_matches/total_samples
            edit_distance = total_dist/total_len
            edit_distance = 1.0 - edit_distance
            print("bleu_score: {:.4f}".format(bleu_score))
            print("exact_match: {:.4f}".format(exact_match))
            print("edit_distance: {:.4f}".format(edit_distance))

