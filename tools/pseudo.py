import torch
import torch.nn.functional as F


def get_ps_label_acc(pred, threshold, label, t=1):

    logit = torch.softmax(pred / t, dim=1)
    max_probs, ps_label = torch.max(logit, dim=1)
    mask = max_probs.ge(threshold).float()

    right_labels = (ps_label == label).float() * mask
    ps_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

    tgt_loss = (F.cross_entropy(logit, ps_label, reduction="none") * mask).sum() / mask.sum() + 1e-4
    # epsilon = 1e-4, used to get rid of inifity gradient
    if torch.isnan(tgt_loss):
        tgt_loss = 0

    return ps_label_acc, ps_label, mask, tgt_loss


def get_ps_label_acc_la(pred, threshold, label, t=1):

    logit = torch.softmax(pred / t, dim=1)
    max_probs, label_p = torch.max(logit, dim=1)
    mask = max_probs.ge(threshold).float()

    right_labels = (label_p == label).float() * mask
    ps_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

    return logit, label_p, mask, ps_label_acc
