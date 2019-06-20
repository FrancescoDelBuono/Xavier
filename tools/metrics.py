import os
import numpy as np

from tools.utils import read_labels, scan_dir

THRESHOLD = 0.5


def IoU(bb_test, bb_gt):
    # Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def evaluation(gt_dir, pred_dir, skip=False):
    # compute precision and recall given two directories with the labels for the same video
    fp = 0
    fn = 0
    tp = 0

    if not os.path.isdir(gt_dir):
        raise ValueError('gt_dir is not a valid directory')

    if not os.path.isdir(pred_dir):
        raise ValueError('pred_dir is not a valid directory')

    frame_count = 0
    object_count = 0

    # scan the two folders to get the list of label files
    # with one file for frame
    list_pred = scan_dir(pred_dir, ext='.txt')
    list_pred = [os.path.basename(x) for x in list_pred]
    list_gt = scan_dir(gt_dir, ext='.txt')
    list_gt = [os.path.basename(x) for x in list_gt]

    # for each label in the files that we have only in the
    # prediction folder we have a false positive
    fp_file = [item for item in list(list_pred) if item not in list_gt]
    frame_count += len(fp_file)
    for file in fp_file:
        file = os.path.join(pred_dir, file)
        rects = read_labels(file, skip)
        fp += len(rects)

    # for each label in the files that we have only in the
    # ground truth folder we have a false negative
    fn_file = [item for item in list(list_gt) if item not in list_pred]
    frame_count += len(fn_file)
    for file in fn_file:
        file = os.path.join(gt_dir, file)
        rects = read_labels(file, skip)
        fn += len(rects)
        object_count += len(rects)

    files = [item for item in list(list_pred) if item in list_gt]
    frame_count += len(files)

    # for each file for the same frame we compute false positive
    # false negative and true positive
    for file in files:
        file_gt = os.path.join(gt_dir, file)
        file_pred = os.path.join(pred_dir, file)

        rects_gt = read_labels(file_gt, skip)
        object_count += len(rects_gt)

        rects_pred = read_labels(file_pred, skip)

        if len(rects_gt) == 0:
            fp += len(rects_pred)
            continue

        if len(rects_pred) == 0:
            fn += len(rects_gt)
            continue

        H = len(rects_pred)
        W = len(rects_gt)

        D = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                D[i, j] = IoU(rects_pred[i], rects_gt[j])

        usedRows = set()  # used predictions
        usedCols = set()  #  used labels

        for row in range(H):
            rank = D[row].argsort()[::-1]
            for col in rank:
                if D[row, col] < THRESHOLD:
                    continue
                if row in usedRows or col in usedCols:
                    continue
                # correct match between prediction rect and label rect
                tp += 1

                usedRows.add(row)
                usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)  # unused prediction are false positive
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)  #  unused labels are false negative

        fp += len(unusedRows)
        fn += len(unusedCols)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print('object count: {}'.format(object_count))
    print('tp: {}'.format(tp))
    print('fp: {}'.format(fp))
    print('fn: {}'.format(fn))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1: {}'.format(f1))

    return precision, recall, f1


def evaluation_2bbox(rects_pred, rects_gt):
    # compute true positive, false positive and false negative
    # given two list of rects using IoU

    tp = 0
    fp = 0
    fn = 0

    if len(rects_gt) == 0:
        fp = len(rects_pred)
    elif len(rects_pred) == 0:
        fn = len(rects_gt)
    else:
        H = len(rects_pred)
        W = len(rects_gt)

        D = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                D[i, j] = IoU(rects_pred[i], rects_gt[j])

        usedRows = set()  # used predictions
        usedCols = set()  #  used labels

        for row in range(H):
            rank = D[row].argsort()[::-1]
            for col in rank:
                if D[row, col] < THRESHOLD:
                    continue
                if row in usedRows or col in usedCols:
                    continue
                tp += 1

                usedRows.add(row)
                usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)  # unused prediction are false positive
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)  #  unused labels are false negative

        fp = len(unusedRows)
        fn = len(unusedCols)

    return tp, fp, fn


def match_pred_gt(rects_pred, rects_gt):
    # function to find for each predicted rect the ground truth rect that
    # provide the highest IoU, where for each ground truth
    # only one predicted rect could be used
    overlapping = [0] * len(rects_pred)

    if len(rects_gt) == 0:
        return overlapping

    elif len(rects_pred) == 0:
        return []

    # compute IoU for each input predicted
    # rect and ground truth rect
    H = len(rects_pred)
    W = len(rects_gt)
    D = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            D[i, j] = IoU(rects_pred[i], rects_gt[j])

    usedRows = set()  # used predictions
    usedCols = set()  #  used labels

    for row in range(H):
        rank = D[row].argsort()[::-1]
        for col in rank:
            if D[row, col] < THRESHOLD:
                continue
            if row in usedRows or col in usedCols:
                continue

            overlapping[row] = D[row, col]

            usedRows.add(row)
            usedCols.add(col)

    return overlapping
