import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
import nltk


def cal_loss(logits, targets, pad_token):
    """args:
        nll loss function used for training
        logits: probability distribution returned by model
                [B, L, V], where L and V are max_len and vocab_size respectively
        targets: target formulas
                [B, L]
    """
    #masking out pad token
    padding = torch.ones_like(targets) * pad_token
    mask = (targets != padding)
    #ensuring first pad token isn't masked out as it is used as an eos token
    #without this step, the model will not learn to generate the eos token
    loc_eos_tokens = torch.argmin(targets, dim=1)
    for i, j in enumerate(loc_eos_tokens):
        mask[i, j] = True
    targets = targets.masked_select(mask)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)
    loss = F.nll_loss(logits, targets)
    return loss


def score(hypotheses, references, pad_token):
    """
    called to get metrics for model performance (bleu score, % exact match, edit distance)
    :param hypotheses: [B, L1] tensor containing predicted sequences
    :param references: [B, L2] tensor containing ground truth sequences
    :param pad_token:
    :return:
    """
    hypotheses = hypotheses.squeeze()
    hypotheses = [hypotheses[i] for i in range(hypotheses.shape[0])]
    #masking out pad token from both input tensors and turning each tensor into a list of lists
    #to ensure compatibility with library functions
    for index, hyp in enumerate(hypotheses):
        mask = torch.ones_like(hyp) * pad_token
        mask = (hyp != mask)
        hypotheses[index] = torch.masked_select(hyp, mask).tolist()
    references = [references[i] for i in range(references.shape[0])]
    for index, ref in enumerate(references):
        mask = torch.ones_like(ref) * pad_token
        mask = (ref != mask)
        references[index] = [torch.masked_select(ref, mask).tolist()]
    bleu = bleu_score(hypotheses, references)
    match = exact_match(hypotheses, references)
    dist, length = edit_distance(hypotheses, references)
    return bleu, match, dist, length


def bleu_score(hypotheses, references):
    return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)


def exact_match(hypotheses, references):
    match = 0
    for hyp, ref in zip(hypotheses, references):
        ref = ref[0]
        match += (hyp == ref)
    return match


def edit_distance(hypotheses, references):
    dist = 0
    length = 0
    for hyp, ref in zip(hypotheses, references):
        ref = ref[0]
        dist += torchaudio.functional.edit_distance(hyp, ref)
        length += max(len(hyp), len(ref))
    #both dist and length are used later to calculate the actual edit distance score
    return dist, length
