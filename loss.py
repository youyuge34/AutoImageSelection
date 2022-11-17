import io
import oss2
import PIL
import numpy as np
from PIL import Image, ImageFile
import torch
import threading


def loss_fn_rank(config, score_1, score_2, gt_labels):
    """
    MarginRankingLoss
    :param score_1:
    :param score_2:
    :return:
    """
    # score_1 = torch.squeeze(score_1)
    # score_2 = torch.squeeze(score_2)
    # label = torch.ones(score_1.shape[0], dtype=torch.int)
    neg = torch.tensor([-1.] * len(gt_labels), dtype=torch.float).cuda()
    rank_loss = config.get_rank_loss()
    gt_labels = torch.where(gt_labels > 0.5, gt_labels, neg)
    out1 = rank_loss(score_1, score_2, gt_labels)
    return out1, gt_labels


criterion = torch.nn.BCELoss()


def rank_sigmoid_loss(score_1, score_2, gt_labels):
    """
        点击率排序 loss， pairwise loss。 点击率高的图片 score 应该高于 ctr低的。 比较相对值的差异 而非 绝对值。
    https://lileicc.github.io/pubs/zhao2019what.pdf
    :param score1:
    :param score2:
    :return:
    """
    # score_1 = torch.squeeze(score_1)
    # score_2 = torch.squeeze(score_2)
    sigma = torch.sigmoid(score_1 - score_2)
    # label = torch.ones(score_1.shape[0], dtype=torch.int)
    # print(sigma.size(), gt_labels.size())
    return criterion(sigma, gt_labels)
