import torch
import torch.nn as nn
from NCEAverage import NCEAverage
from utils import l2_normalize


eps = 1e-7

class NCECriterion(nn.Module):
    def __init__(self, len_neg):
        super(NCECriterion, self).__init__()
        self.num_data = len_neg

    def forward(self, x):
        """
        :param x: output matrix with size (batch_size, k+1). Each element is (exp(torch.mm(vi, v.t()))/tau)/Zi which is P(1|vi,normal_v)
        :return: NCE Loss
        """
        batch_size = x.size(0)
        k = x.size(1) - 1  # K is the number of negative samples

        # Assume noise distribution is a uniform distribution
        q_noise = 1. / self.num_data

        # P(1|normal_v, vi) = p_p / (p_p + k*q_noise)
        p_p = x.select(1, 0)  # equal to x[:, 0] and p_p is p(vi|normal_v)
        log_D1 = torch.div(p_p, p_p.add(k * q_noise + eps)).log_()

        # Second term of NCE Loss which is loss for negative pairs
        # P(0|normal_v, vi_prime) = P(origin=noise) = k*q_noise / (p_n + k*q_noise)
        p_n = x.narrow(1, 1, k)  # narrow(dim, start, len) equal to x[:, 1:K+1] and p_n is p(vi_prime|normal_v)
        log_D0 = torch.div(p_n.clone().fill_(k*q_noise), p_n.add(k*q_noise+eps)).log_()  # clone is just to get a same size matrix and be filled with  k*q_noise

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / batch_size



        return loss

if __name__ == '__main__':
    average = NCEAverage(128, 9000, 0.07, 28, 0.9).cuda()
    criterion = NCECriterion(9000)
    dummy_n_embeddings = torch.randn(4, 128).cuda()
    dummy_a_embeddings = torch.randn(28, 128).cuda()
    dummy_n_embeddings = l2_normalize(dummy_n_embeddings)
    dummy_a_embeddings = l2_normalize(dummy_a_embeddings)

    outs, probs = average(dummy_n_embeddings, dummy_a_embeddings)
    print(outs)
    print(outs.shape)
    loss = criterion(outs)
    print(loss)