import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = []
    for n in cls_num_list:
        per_cls_weights.append((1-beta)/(1-beta**n))
    per_cls_weights = torch.Tensor(per_cls_weights).cuda()
    per_cls_weights = per_cls_weights / torch.sum(per_cls_weights)
    return per_cls_weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        cross_entropy_vector = F.cross_entropy(input, F.one_hot(target, len(self.weight)).float(), reduction="none")
        softmax_probs = F.softmax(input, dim=1)
        focal_factor = torch.pow(1-(softmax_probs*F.one_hot(target,len(self.weight))).sum(1),self.gamma)
        focal_loss = focal_factor * cross_entropy_vector
        class_balanced_focal_loss = self.weight[target] * focal_loss


        return class_balanced_focal_loss.sum()