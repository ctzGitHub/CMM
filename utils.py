# -*- coding: utf-8 -*-
'''
@ author:
'''

import time
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score, accuracy_score, \
    recall_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[1])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = roc_auc_score(y_true, y_pred)
    return auc, class_auc

def apply_thresholds(preds, threshold):
    tmp = []
    for p in preds:
        tmp_p = (p > threshold).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return f1_score(y_true, y_pred_binary, average="samples")


def apply_thresholds(preds, threshold):
    tmp = []
    for p in preds:
        tmp_p = (p > threshold).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp

# PRINT TIME
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)


def cal_accuracy_score(y_true, y_pre,threshold=0.5):
    # y_true = y_true.cpu().view(-1)
    # y_pre =y_pre.cpu().view(-1,5)
    # print(y_pre.shape)
    # print(y_true.shape)
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_pre,y_true)

def compute_mAP(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i], average="samples"))
    return np.mean(AP)

def calc_recall(y_true, y_pre, threshold=0.5):
    y_true = np.array(y_true)
    y_pre = np.array(y_pre)
    y_pred_binary = apply_thresholds(y_pre, threshold)
    return recall_score(y_true, y_pred_binary, average="samples")



# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss