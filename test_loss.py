import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from L_GM_loss import LGMLoss

# demo funtion loss, comes from author's tf version:
# https://github.com/WeitaoVan/L-GM-loss/tree/master/tensorflow

def tc_lgm_logits(feat, num_classes, labels=None, alpha=0.1, lambda_=0.01, means=None):
    N= feat.size()[0]
    feat_len = feat.size()[1]

    XY = torch.matmul(feat, torch.transpose(means, 0, 1))
    XX = torch.sum(feat ** 2, dim=1, keepdim=True)
    YY = torch.sum(torch.transpose(means, 0, 1)**2, dim=0, keepdim=True)
    neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

    if labels is None:
        psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
        means_batch = torch.index_select(means, dim=0, index=psudo_labels)
        likelihood_reg_loss = lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / N)
        return neg_sqr_dist, likelihood_reg_loss, means

    label = labels.view(labels.size()[0], -1)
    ALPHA = torch.zeros(N, num_classes).scatter_(1, label, alpha)

    K = ALPHA + torch.ones([N, num_classes])
    logits_with_margin = torch.mul(neg_sqr_dist, K)
    means_batch = torch.index_select(means, dim=0, index=labels)
    likelihood_reg_loss = lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / N)
    return neg_sqr_dist, likelihood_reg_loss, means

if __name__ == '__main__':
    num_classes = 5
    num_data = 2
    feat_dim = 5
    num_classes = 3

    np_feat = np.random.randn(num_data, feat_dim).astype(np.float32)
    np_labels = np.random.randint(0, num_classes, size=num_data).astype(np.int64)


    lgmloss = LGMLoss(num_classes, feat_dim, alpha=1.0, lambda_=1.0).cuda()

    # this optim 'optimzer4lgm' use to update lgmloss param: 'means',
    # You still need to build a optim for model.
    # for example:
    # optimzer = optim.SGD(model.parameters(), lr=0.01)
    optimzer4lgm = optim.SGD(lgmloss.parameters(), lr=0.1)

    # Simulate two iterations
    for _ in range(2):
        tc_feat = torch.tensor(np_feat).cuda()
        tc_labels = torch.tensor(np_labels).cuda()

        _, loss, _ = lgmloss(tc_feat, tc_labels)
        print(loss)

        _, tc_loss, _ = tc_lgm_logits(tc_feat.cpu(), num_classes,
                                     labels=tc_labels.cpu(),
                                     alpha=1.0,
                                     lambda_=1.0,
                                     means=lgmloss.means.cpu())
        print(loss)
        print(tc_loss)

        print('--'*10)

        #here also need optimzer for model:
        #for example:
        #optimzer.zero_grad()
        optimzer4lgm.zero_grad()
        loss.backward()

        # print(lgmloss.means.grad)
        print(lgmloss.means)
        print('--'*10)

        #here also need optimzer apply grad:
        #for example:
        #optimzer.step()
        optimzer4lgm.step()
