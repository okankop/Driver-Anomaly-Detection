import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    calculate Contrastive Loss for Contrastive Learning
    n_vec: normalized vectors from normal driving video
    a_vec: normalized vectors from anormal driving video
    tau: is a temperature parameter that controls the concentration level of the distribution of embedded vectors.
    :return: Contrastive Loss
    """
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, n_vec, a_vec, tau):

        n_scores = torch.mm(n_vec, n_vec.t())
        n_scores = n_scores[~torch.eye(n_scores.shape[0], dtype=bool)].reshape(n_vec.shape[0], -1).div_(tau).exp_().view(-1, 1)  # n_scores is numerator
        print(f'normal data similarity: {n_scores.shape}')
        print(n_scores)
        print('\n')

        n_a_scores = torch.mm(n_vec, a_vec.t()).div_(tau).exp()
        print(f'anormal data similarity: {n_a_scores.shape}')
        print(n_a_scores)
        print('\n')

        sum_n_a = torch.sum(n_a_scores, dim=1, keepdim=True)  # sum term of denominator
        print(f'sum n_a: {sum_n_a.shape}')
        print(sum_n_a)
        print('\n')

        sum_n_a = sum_n_a.repeat(1, (n_vec.shape[0]-1)).view(-1, 1)  # repeat sum term for the number of normal samples times
        print(f'sum_n_a repeat: {sum_n_a.shape}')
        print(sum_n_a)
        print('\n')

        denominator = n_scores + sum_n_a
        print(f'denominator: {denominator.shape}')
        print(denominator)
        print('\n')
        p = torch.log(torch.div(n_scores, denominator))
        print(f'p: {p}')
        loss = -torch.sum(p)
        return loss


