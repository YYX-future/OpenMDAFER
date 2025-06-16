import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class AccuracyCounter:
    def __init__(self, length, open_class):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length
        self.open_class = open_class

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def add_correct_list(self, N_correct):
        self.Ncorrect = N_correct

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def h_score(self, open_class):
        self.clear_zero()
        correct = np.sum(self.Ncorrect[:open_class]) + np.sum(self.Ncorrect[open_class + 1:])
        total = np.sum(self.Ntotal[:open_class]) + np.sum(self.Ntotal[open_class + 1:])
        common_acc = correct / total
        print(f'Ncorrect: {self.Ncorrect}')
        open_acc = self.Ncorrect[open_class] / self.Ntotal[open_class]
        return 2 * common_acc * open_acc / (common_acc + open_acc),  common_acc, self.Ncorrect, self.Ntotal


def get_confidence_three_prediction(fc1_s, fc2_s, fc2_s2):

    fc1_s = torch.nn.Softmax(-1)(fc1_s)
    fc2_s = torch.nn.Softmax(-1)(fc2_s)
    fc2_s2 = torch.nn.Softmax(-1)(fc2_s2)

    conf_1, indice_1 = torch.max(fc1_s, 1)
    conf_2, indice_2 = torch.max(fc2_s, 1)
    conf_3, indice_3 = torch.max(fc2_s2, 1)

    confidence = (conf_1 + conf_2 + conf_3) / 3

    return confidence, indice_1, indice_2, indice_3


def get_confidence_one_prediction(fc1_s):

    fc1_s = torch.nn.Softmax(-1)(fc1_s)

    conf_1, indice_1 = torch.max(fc1_s, 1)
    confidence = conf_1

    return confidence, indice_1


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()


def get_threshold(sft_scores):

    thresholds = torch.cat(sft_scores)
    thd_min = torch.min(thresholds)
    thd_max = torch.max(thresholds)
    threshold_range_list = [thd_min + (thd_max - thd_min) * i / 10 for i in range(10)]
    unknown_threshold = sum(threshold_range_list) / len(threshold_range_list)
    return unknown_threshold


def get_three_counters(num_classes, unknown_threshold,
                 p_label1,  p_label2,  p_label3, all_labels, all_score, pro_confidence, open_class):

    counters1 = AccuracyCounter(num_classes, open_class)
    counters2 = AccuracyCounter(num_classes, open_class)
    counters3 = AccuracyCounter(num_classes, open_class)

    known_scores, unknown_scores = [], []

    for (each_p1, each_p2, each_p3, each_label, score, pro) in \
            zip(p_label1, p_label2, p_label3, all_labels, all_score, pro_confidence):

        if each_label != open_class:
            counters1.add_total(each_label)
            counters2.add_total(each_label)
            counters3.add_total(each_label)

            known_scores.append(1 - pro.item())

            if score.item() >= unknown_threshold:
                if each_p1 == each_label:
                    counters1.add_correct(each_label)
                if each_p2 == each_label:
                    counters2.add_correct(each_label)
                if each_p3 == each_label:
                    counters3.add_correct(each_label)

        else:
            counters1.add_total(open_class)
            counters2.add_total(open_class)
            counters3.add_total(open_class)
            unknown_scores.append(1 - pro.item())
            if score.item() < unknown_threshold:
                counters1.add_correct(open_class)
                counters2.add_correct(open_class)
                counters3.add_correct(open_class)
    return counters1, counters2, counters3, known_scores, unknown_scores


def get_one_counters(num_classes, unknown_threshold, p_label, all_labels, all_score, pro_confidence, open_class):

    counters = AccuracyCounter(num_classes, open_class)

    known_scores, unknown_scores = [], []

    for (each_p1, each_label, score, pro) in zip(p_label, all_labels, all_score, pro_confidence):
        if each_label != open_class:
            if each_label > open_class:
                each_label -= 1
            counters.add_total(each_label)

            known_scores.append(1 - pro.item())

            if score.item() >= unknown_threshold:
                if each_p1 - 1 == each_label:
                    counters.add_correct(each_label)
        else:
            counters.add_total(-1)
            unknown_scores.append(1 - pro.item())
            if score.item() < unknown_threshold:
                counters.add_correct(-1)

    return counters, known_scores, unknown_scores


def calculate_auc(counters, known_scores, unknown_scores, open_class):

    num_zeros = int(np.sum(counters.Ntotal[:open_class]) + np.sum(counters.Ntotal[open_class + 1:]))
    num_ones = int(counters.Ntotal[open_class])

    y_true = np.array([0] * num_zeros + [1] * num_ones)

    y_score = np.concatenate([known_scores, unknown_scores])

    return roc_auc_score(y_true, y_score)
