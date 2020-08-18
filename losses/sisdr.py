"""!
@brief SISNR very efficient computation in Torch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign

"""

import torch
import torch.nn as nn
import itertools


def _sdr( y, z, SI=False):
    if SI:
        a = ((z*y).mean(-1) / (y*y).mean(-1)).unsqueeze(-1) * y
        return 10*torch.log10( (a**2).mean(-1) / ((a-z)**2).mean(-1))
    else:
        return 10*torch.log10( (y*y).mean(-1) / ((y-z)**2).mean(-1))

# Negative SDRi loss
def sdri_loss( y, z, of=0):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    s = _sdr( y, z, SI=False) - of
    return -s.mean()

# Negative SI-SDRi loss
def sisdr_loss( y, z, of=0):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    s = _sdr( y, z, SI=True) - of
    return -s.mean()

# Negative PIT loss
def pit_loss( y, z, of=0, SI=False):
    # Add a batch dimension if it's missing
    if len( y.shape) < 3:
        y = y.unsqueeze(0)
    if len( z.shape) < 3:
        z = z.unsqueeze(0)

    # Get all possible target source permutation SDRs and stack them
    p = list( itertools.permutations( range( y.shape[-2])))
    s = torch.stack( [_sdr( y[:,j,:], z, SI) for j in p], dim=2)

    # Get source-average SDRi
    # s = (s - of.unsqueeze(2)).mean(1)
    s = s.mean(1)

    # Find and return permutation with highest SDRi (negate since we are minimizing)
    i = s.argmax(-1)
    j = torch.arange( s.shape[0], dtype=torch.long, device=i.device)
    return -s[j,i].mean()


class PermInvariantSISDR(nn.Module):
    """!
    Class for SISDR computation between reconstructed signals and
    target wavs by also regulating it with learned target masks."""

    def __init__(self,
                 zero_mean=False,
                 n_sources=2,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False,
                 pit=False,
                 or_pit=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.improvement = improvement
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results
        self.pit=pit
        self.or_pit=or_pit
        self.best_perm = None
        self.best_target = None


    def normalize_input(self, pr_batch, t_batch, initial_mixtures=None):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        if initial_mixtures is not None:
            min_len = min(min_len, initial_mixtures.shape[-1])
            initial_mixtures = initial_mixtures[:, :, :min_len]
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)
            if initial_mixtures is not None:
                initial_mixtures = initial_mixtures - torch.mean(
                    initial_mixtures, dim=-1, keepdim=True)
        return pr_batch, t_batch, initial_mixtures

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_sisnrs(self,
                                permuted_pr_batch,
                                t_batch,
                                t_t_diag, eps=10e-8):
        s_t = (self.dot(permuted_pr_batch, t_batch) /
               (t_t_diag + eps) * t_batch)
        e_t = permuted_pr_batch - s_t
        sisnrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                  (self.dot(e_t, e_t) + eps))
        return sisnrs

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      initial_mixtures=None,
                      eps=10e-8):

        t_t_diag = self.dot(t_batch, t_batch)

        self.best_perm = []
        if self.pit:
            sisnr_l = []
            for perm in self.permutations:
                if pr_batch.shape[1] > t_batch.shape[1]:
                    # when pr_batch contains 2 est speeches and 1 noise, t_batch only contain 2 clean speeches
                    permuted_pr_batch = pr_batch[:, perm[:t_batch.shape[1]], :]
                else:
                    permuted_pr_batch = pr_batch[:, perm, :]
                sisnr = self.compute_permuted_sisnrs(permuted_pr_batch,
                                                     t_batch,
                                                     t_t_diag, eps=eps)
                sisnr_l.append(sisnr)
            all_sisnrs = torch.cat(sisnr_l, -1)
            best_sisdr = torch.max(all_sisnrs.mean(-2), -1)[0]
            best_perm = torch.max(all_sisnrs.mean(-2), -1)[1]
            for i in range(best_perm.shape[0]):
                self.best_perm.append(self.permutations[best_perm[i]])
        elif self.or_pit:
            sisnr_l = []
            for i in range(t_batch.shape[1]):
                rest_ind = list(range(t_batch.shape[1]))
                rest_ind.pop(i)
                target_t_batch = torch.cat([t_batch[:, i, :].unsqueeze(1), t_batch[:, rest_ind, :].sum(dim=1, keepdim=True)], dim=1)
                sisnr = self.compute_permuted_sisnrs(pr_batch,
                                                     target_t_batch,
                                                     self.dot(target_t_batch, target_t_batch), eps=eps)
                sisnr_l.append(sisnr)
            all_sisnrs = torch.cat(sisnr_l, -1)
            best_sisdr = torch.max(all_sisnrs.mean(-2), -1)[0]
            best_target = torch.max(all_sisnrs.mean(-2), -1)[1]
            # print(best_target)
            self.best_target = best_target
            # self.best_perm = []
            # for i in range(best_perm.shape[0]):
            #     self.best_perm.append(self.permutations[best_perm[i]])
        else:
            if pr_batch.shape[1] > t_batch.shape[1]:
                pr_batch = pr_batch[:, :t_batch.shape[1], :]
            best_sisdr = self.compute_permuted_sisnrs(pr_batch,
                                                 t_batch,
                                                 t_t_diag, eps=eps)
            best_sisdr = best_sisdr.mean(dim=1).squeeze(1)
            for i in range(t_batch.shape[0]):
                self.best_perm.append(list(torch.arange(pr_batch.shape[1])))

        if self.improvement:
            initial_mix = initial_mixtures.repeat(1, t_batch.shape[1], 1)
            base_sisdr = self.compute_permuted_sisnrs(initial_mix,
                                                      t_batch,
                                                      t_t_diag, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return - best_sisdr
        return best_sisdr

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                initial_mixtures=None):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.
        :param initial_mixtures: Initial Mixtures for SISDRi: Torch Tensor
                                 of size: batch_size x 1 x length_of_wavs

        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """

        if self.or_pit:
            self.pit = False

        if pr_batch.shape[1] != self.n_sources:
            self.n_sources = pr_batch.shape[1]
            self.permutations = list(itertools.permutations(
                torch.arange(self.n_sources)))

        pr_batch, t_batch, initial_mixtures = self.normalize_input(
            pr_batch, t_batch, initial_mixtures=initial_mixtures)

        sisnr_l = self.compute_sisnr(pr_batch,
                                     t_batch,
                                     eps=eps,
                                     initial_mixtures=initial_mixtures)

        return sisnr_l