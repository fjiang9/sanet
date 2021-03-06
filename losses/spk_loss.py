import torch
import torch.nn.functional as F


from typing import Tuple
from torch import nn, Tensor


def convert_label_to_similarity(feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    normed_feature = F.normalize(feature, p=2, dim=1)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    # print(positive_matrix.int().sum(), negative_matrix.int().sum())
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class SoftmaxLoss(nn.Module):
    def __init__(self, input_size, output_size, normalize=True):
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(input_size, int(output_size), bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, 0.25)
        self.weight = self.fc.weight
        self.normalize = normalize

    def forward(self, x, y):
        if self.normalize:
            self.fc.weight.renorm(2, 0, 1e-5).mul(1e5)
        prob = F.log_softmax(self.fc(x), dim=1)
        self.prob = prob
        loss = F.nll_loss(prob, y)
        return loss


