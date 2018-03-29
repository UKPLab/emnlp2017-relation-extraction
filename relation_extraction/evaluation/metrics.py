# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import numpy as np
import tqdm


def accuracy(prediction_classes, gold_labels):
    acc = len((prediction_classes == gold_labels).nonzero()[0]) / len(gold_labels)
    return acc


def accuracy_per_sentence(predicted_batch, gold_batch, threshold=1.0):
    if len(predicted_batch) != len(gold_batch):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")

    correct = 0
    for i in range(len(gold_batch)):
        rec_batch = accuracy(predicted_batch[i], gold_batch[i])
        if rec_batch >= threshold:
            correct += 1

    acc_batch = correct / float(len(gold_batch))

    return acc_batch


def compute_micro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    if i == -1:
        i = len(predicted_idx)
    if i < len(gold_idx):
        predicted_idx = np.concatenate([predicted_idx[:i], np.ones(len(gold_idx)-i)])
    t = predicted_idx != empty_label
    tp = len((predicted_idx[t] == gold_idx[t]).nonzero()[0])
    tp_fp = len((predicted_idx != empty_label).nonzero()[0])
    tp_fn = len((gold_idx != empty_label).nonzero()[0])
    prec = (tp / tp_fp) if tp_fp != 0 else 1.0
    rec = tp / tp_fn if tp_fp != 0 else 0.0
    f1 = 0.0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {0, empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec+avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def compute_precision_recall_curve(predictions, gold_labels, micro=False, empty_label=0):
    prediction_classes = np.argmax(predictions, axis=1)
    stacked = np.stack([np.max(predictions, axis=1), prediction_classes, gold_labels]).T

    stacked = stacked[stacked[:, 0].argsort()][::-1]
    prec_rec_values = {}
    for i in tqdm.tqdm(range(1, len(stacked), 1000)):
        if micro:
            avg_prec, avg_rec, _ = compute_micro_PRF(stacked[:, 1], stacked[:, 2], i, empty_label=empty_label)
        else:
            avg_prec, avg_rec, _ = compute_macro_PRF(stacked[:, 1], stacked[:, 2], i, empty_label=empty_label)
        prec_rec_values[avg_rec] = avg_prec
    curve = sorted(prec_rec_values.items(), key=lambda el: el[0])[1:]
    return curve


def micro_avg_precision(guessed, correct, empty=None):
    """
    Tests:
    >>> micro_avg_precision(['A', 'A', 'B', 'C'],['A', 'C', 'C', 'C'])
    0.5
    >>> round(micro_avg_precision([0,0,0,1,1,1],[1,0,1,0,1,0]), 6)
    0.333333
    """
    correctCount = 0
    count = 0
    
    idx = 0
    while idx < len(guessed):
        if guessed[idx] != empty:
            count += 1
            if guessed[idx] == correct[idx]:
                correctCount += 1
        idx += 1
    precision = 0
    if count > 0:    
        precision = correctCount / count
        
    return precision


if __name__ == "__main__":
    # Testing
    import doctest
    print(doctest.testmod())
