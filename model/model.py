import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from positional_encoding import positionalencoding2d

class Model(nn.Module):
    def __init__(self, device, vocab_size, embedding_size, dropout=0):
        super(Model, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1), 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        #calc initial states for LSTM
        self.calc_h0 = nn.Linear(512, 512)
        self.calc_c0 = nn.Linear(512, 512)
        self.calc_o0 = nn.Linear(512, 512)

        self.decoder = nn.LSTMCell(embedding_size+512, 512)
        self.W_3 = nn.Linear(512+512, 512, bias=False)
        self.calc_logits = nn.Linear(512, vocab_size, bias=False)

        #attention
        self.attention_w2_h = nn.Linear(512, 512, bias=False)
        self.attention_w1_e = nn.Linear(512, 512, bias=False)
        self.attention_beta = nn.Linear(512, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def pos_encode(self, conv_output):
        """
        adds 2D positional encoding to encoder output and flattens and reshapes resulting tensor
        :param conv_output: [B,512,H',W']
        :return: flat_conv_output [B, H'*W', 512]
        """
        conv_output += positionalencoding2d(self.device, 512, conv_output.size(dim=2), conv_output.size(dim=3))
        conv_output = conv_output.permute(0, 2, 3, 1)
        B, H, W, _ = conv_output.shape
        conv_output = conv_output.contiguous().view(B, H*W, -1)
        return conv_output

    def encode(self, batch_images):
        """
        :param batch_images: [B, C, H, W]
        :return: flattened output of convolutional layer with positional encoding
        information added, [B, H'*W', 512]
        """
        encoded_images = self.encoder(batch_images)
        return self.pos_encode(encoded_images)

    def get_context(self, flat_conv_output, curr_hidden_state):
        """
        :param flat_conv_output: [B, H'*W', 512]
        :param curr_hidden_state: [B, 512]
        :return: context vector, [B, 512]
        """
        weights = self.attention_w1_e(flat_conv_output) + self.attention_w2_h(curr_hidden_state).unsqueeze(1)
        weights = F.tanh(weights)
        weights = self.attention_beta(weights)
        weights = weights.squeeze(2)
        weights = F.softmax(weights, dim=1)
        context = torch.bmm(weights.unsqueeze(1), flat_conv_output)
        return context.squeeze(1)

    def get_h_0(self, mean_flat_conv_output):
        """
        :param mean_flat_conv_output: [B, 512]
        :return: initial h_t for LSTM, [B, 512]
        """
        return F.tanh(self.calc_h0(mean_flat_conv_output))

    def get_c_0(self, mean_flat_conv_output):
        """
        :param mean_flat_conv_output: [B, 512]
        :return: initial c_t for LSTM, [B, 512]
        """
        return F.tanh(self.calc_c0(mean_flat_conv_output))

    def get_o_0(self, mean_flat_conv_output):
        """
        :param mean_flat_conv_output: [B, 512]
        :return: initial o_t for LSTM, [B, 512]
        """
        return F.tanh(self.calc_o0(mean_flat_conv_output))

    def get_init_states(self, flat_conv_output):
        """
        :param flat_conv_output: [B, H'*W', 512]
        :return: initial h, c, t for LSTM, all [B, 512]
        """
        mean_flat_conv_output = flat_conv_output.mean(dim=1)
        h_0 = self.get_h_0(mean_flat_conv_output)
        c_0 = self.get_c_0(mean_flat_conv_output)
        o_0 = self.get_o_0(mean_flat_conv_output)
        return h_0, c_0, o_0

    def get_o_t(self, context_t, h_t):
        """
        :param context_t: context vector at time t, [B, 512]
        :param h_t: [B, 512]
        :return: o_t [B, 512]
        """
        o_t = self.W_3(torch.cat((h_t, context_t), dim=1))
        o_t = F.tanh(o_t)
        return o_t

    def decode_t(self, h_t, c_t, o_t, prev_token, flat_conv_output):
        """
        generate one timestep of LSTM output
        :param h_t: [B, 512]
        :param c_t: [B, 512]
        :param o_t: [B, 512]
        :param prev_token: [B, 1]
        :param flat_conv_output: [B, H'*W', 512]
        :return: new states for LSTM, probabilities of next tokens
        """
        embedded_prev_token = self.embedding(prev_token).squeeze(1)
        rnn_input = torch.cat((embedded_prev_token, o_t), dim=1)
        h_new, c_new = self.decoder(rnn_input, (h_t, c_t))
        h_new = self.dropout(h_new)
        c_new = self.dropout(c_new)
        context_new = self.get_context(flat_conv_output, h_new)
        o_new = self.get_o_t(context_new, h_new)
        o_new = self.dropout(o_new)
        distribution = F.softmax(self.calc_logits(o_new), dim=1)
        return h_new, c_new, o_new, distribution

    def forward(self, batch_images, formulas):
        """
        feeds in ground truth previous token at each timestep,
        only used for training, as ground truth is required
        :param batch_images: [B, C, H, W]
        :param formulas: [B, formula_len]
        :return: logits, [B, formula_len, vocab_size]
        """
        flat_conv_output = self.encode(batch_images)
        h_t, c_t, o_t = self.get_init_states(flat_conv_output)
        #subtract 1 because model won't generate start token
        formula_len = formulas.size(1) - 1
        logits = []
        for t in range(formula_len):
            ground_truth = formulas[:,t:t+1]
            h_t, c_t, o_t, curr_logit = self.decode_t(h_t, c_t, o_t, ground_truth, flat_conv_output)
            logits.append(curr_logit)
        logits = torch.stack(logits, dim=1)
        return logits

