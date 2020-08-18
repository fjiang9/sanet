import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import glob2
import numpy as np
from models.model_base import Model_base
from torch_stft import STFT


class Encoder1D(nn.Module):
    '''
    A 1D convolutional block that transforms signal in
    wave form into higher dimension

    input shape: [batch, 1, n_samples]
    output shape: [batch, channels, n_samples//win_len]

    channels: number of output frequencies for the encoding convolution
    win_len: int, length of the encoding filter
    '''

    def __init__(self, channels=64, win_len=16, hop_size=8, enc_act='relu', bias=True, stft=None):
        super().__init__()
        self.stft = stft
        if self.stft is None:
            self.conv = nn.Conv1d(1,
                                  channels,
                                  kernel_size=win_len,
                                  stride=hop_size,
                                  # padding=win_len // 2,
                                  bias=bias)
            self.enc_act = enc_act

    def forward(self, signal):
        if self.stft is not None:
            magnitude, phase = self.stft.transform(signal)
            return (magnitude, phase)
        if self.enc_act == 'relu':
            return F.relu(self.conv(signal))
        elif self.enc_act == 'linear':
            return self.conv(signal)
        if self.enc_act == 'sigmoid':
            return torch.sigmoid(self.conv(signal))
        else:
            raise ValueError('Encoder activation function must be linear, relu, or sigmoid')


class Decoder1D(nn.Module):
    '''
    A 1D deconvolutional block that transforms
    encoded representation into wave form

    input shape: [batch, channels, win_len]
    output shape: [batch, 1, win_len*n_samples]

    channels: number of output frequencies for the encoding convolution
    win_len: length of the encoding filter
    '''

    def __init__(self, channels=64, win_len=16, hop_size=8, bias=True, stft=None):
        super().__init__()
        self.stft = stft
        if self.stft is None:
            self.deconv = nn.ConvTranspose1d(channels,  # channels*n_sources
                                             1,  # n_sources
                                             kernel_size=win_len,
                                             # padding=win_len // 2,
                                             stride=hop_size,
                                             # groups=n_sources,
                                             # output_padding=(win_len // 2) - 1,
                                             bias=bias)


    def forward(self, x, phase=None):
        if self.stft is not None:
            output = self.stft.inverse(x, phase).unsqueeze(1)
            return output
        return self.deconv(x)


class CAE(Model_base):
    '''
        Adaptive basis encoder
        channels: The number of frequency like representations
        win_len: The number of samples in kernel 1D convolutions

    '''

    def __init__(self, model_args=None):
        super().__init__()

        self.model_args = model_args
        self.hop_size = model_args['stride']
        self.channels = model_args['n_filters']
        self.win_len = model_args['kernel_size']
        self.bias = model_args['bias']
        self.enc_act = model_args['enc_act']
        self.regularizer = model_args['mask']

        if 'use_stft' not in model_args.keys():
            self.use_stft = None
        else:
            self.use_stft = model_args['stft']
        if self.use_stft:
            self.stft = STFT(filter_length=(self.channels - 1) * 2,
                             hop_length=self.hop_size,
                             win_length=self.win_len,
                             window='hann')
            self.encoder = Encoder1D(stft=self.stft)
            self.decoder = Decoder1D(stft=self.stft)
        else:
            self.encoder = Encoder1D(self.channels, self.win_len, self.hop_size, enc_act=self.enc_act, bias=self.bias)
            self.decoder = Decoder1D(self.channels, self.win_len, self.hop_size, bias=self.bias)

        self.eps = 10e-9

    def get_target_masks(self, clean_sources):
        if self.use_stft:
            enc_masks = self.encoder(clean_sources[:, 0, :].unsqueeze(1))[0].unsqueeze(1)
            for i in range(clean_sources.shape[1] - 1):
                enc_masks = torch.cat(
                    (enc_masks, self.encoder(clean_sources[:, i + 1, :].unsqueeze(1))[0].unsqueeze(1)),
                    dim=1)
        else:
            enc_masks = self.encoder(clean_sources[:, 0, :].unsqueeze(1)).unsqueeze(1)
            for i in range(clean_sources.shape[1] - 1):
                enc_masks = torch.cat(
                    (enc_masks, self.encoder(clean_sources[:, i + 1, :].unsqueeze(1)).unsqueeze(1)),
                    dim=1)
        if clean_sources.shape[1] == 1:
            return torch.ones(enc_masks.shape).cuda()
        if self.regularizer == 'irm':
            total_mask = torch.sum(enc_masks, dim=1, keepdim=True)
            enc_masks /= (total_mask + self.eps)
        elif self.regularizer == 'softmax':
            enc_masks = F.softmax(enc_masks, dim=1)
        elif self.regularizer == 'wfm':
            total_mask = torch.sum(enc_masks.pow(2), dim=1, keepdim=True)
            enc_masks = enc_masks.pow(2) / (total_mask + self.eps)
        else:
            raise NotImplementedError(
                "Regularizer: {} is not implemented".format(self.regularizer))
        return enc_masks

    def get_encoded_sources(self, mixture, clean_sources, enc_masks):
        enc_mixture = self.encoder(mixture)
        enc_sources = enc_masks * enc_mixture.unsqueeze(1)
        if self.deep:
            if self.pinv_weights:
                def dec2v(x):
                    U = self.encoder.encoding[1].weight.data.squeeze(2)
                    return torch.bmm(U.pinverse().unsqueeze(0).expand(x.shape[0], -1, -1), x)
            else:
                dec2v = self.decoder.decoding
            enc_sources_v = F.relu(self.encoder.conv(clean_sources[:, 0, :].unsqueeze(1)).unsqueeze(1))
            dec_sources_v = dec2v(enc_sources[:, 0, :, :]).unsqueeze(1)
            for i in range(enc_masks.shape[1] - 1):
                enc_sources_v = torch.cat(
                    (enc_sources_v, F.relu(self.encoder.conv(clean_sources[:, i + 1, :].unsqueeze(1)).unsqueeze(1))),
                    dim=1)
                dec_sources_v = torch.cat((dec_sources_v, dec2v(enc_sources[:, i + 1, :, :]).unsqueeze(1)), dim=1)
            return enc_sources, enc_sources_v, dec_sources_v
        else:
            return enc_sources

    def get_enc_weight(self, enc_mixture, thr=0.9):
        # input (B, F, T)
        # output (B, 1, F*T)
        enc_mixture = enc_mixture.contiguous().view(enc_mixture.shape[0], -1)
        weight = torch.zeros(enc_mixture.shape)
        _, ind = torch.topk(enc_mixture, int(np.round(thr * enc_mixture.shape[1])))
        for i in range(weight.shape[0]):
            weight[i, ind[i]] = 1
        return weight.unsqueeze(1)

    def get_rec_sources(self, enc_masks, enc_mixture, phase=None):
        # enc_masks.shape = (B, C, F, T)
        # enc_mixture.shape = (B, F, T)
        est_sources = None
        for i in range(enc_masks.shape[1]):
            enc = enc_masks[:, i, :, :] * enc_mixture
            est_source_temp = self.decoder(enc, phase=phase)
            if i == 0:
                est_sources = est_source_temp
            else:
                est_sources = torch.cat((est_sources, est_source_temp), dim=1)
        return est_sources

    def forward(self, mixture, clean_sources):
        if self.use_stft:
            enc_mixture = self.encoder(mixture)
            phase = enc_mixture[1]
            enc_mixture = enc_mixture[0]
        else:
            enc_mixture = self.encoder(mixture)
            phase = None
        enc_masks = self.get_target_masks(clean_sources)
        if self.regularizer == 'linear':
            est_source1 = self.decoder(enc_masks[0] * self.encoder(clean_sources[:, 0, :].unsqueeze(1)))
            est_source2 = self.decoder(enc_masks[1] * self.encoder(clean_sources[:, 1, :].unsqueeze(1)))
            est_sources = torch.cat((est_source1, est_source2), dim=1)
        else:
            est_sources = self.get_rec_sources(enc_masks, enc_mixture, phase=phase)
        return est_sources, enc_masks, enc_mixture

