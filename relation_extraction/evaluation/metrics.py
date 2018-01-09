# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#


def evaluate_batch_based(predicted_batch, gold_batch, threshold = 1.0, idx2label=None, empty_label = None):
    if len(predicted_batch) != len(gold_batch):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")

    correct = 0
    for i in range(len(gold_batch)):
        rec_batch = micro_avg_precision(predicted_batch[i], gold_batch[i], empty_label)
        if rec_batch >= threshold:
            correct += 1

    acc_batch = correct / float(len(gold_batch))

    return acc_batch


def evaluate_instance_based(predicted_idx, gold_idx, idx2label=None, empty_label = None):
    if len(predicted_idx) != len(gold_idx):
        raise TypeError("predicted_idx and gold_idx should be of the same length.")
    if idx2label:
        label_y = [idx2label[element] for element in gold_idx]
        pred_labels = [idx2label[element] for element in predicted_idx]
    else:
        label_y = gold_idx
        pred_labels = predicted_idx
    
    prec = micro_avg_precision(pred_labels, label_y, empty_label)
    rec = micro_avg_precision(label_y, pred_labels, empty_label)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
        
    return prec, rec, f1    
    

def micro_avg_precision(guessed, correct, empty = None):
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
