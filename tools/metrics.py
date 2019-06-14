import os
import numpy as np
from shapely.geometry import box

from tools.utils import read_labels, scan_dir


def IoU(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
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
    fp = 0
    fn = 0
    tp = 0

    if not os.path.isdir(gt_dir):
        raise ValueError('gt_dir is not a valid directory')

    if not os.path.isdir(pred_dir):
        raise ValueError('pred_dir is not a valid directory')

    iou_tot = 0
    frame_count = 0
    object_count = 0
    th = 0.5

    list_pred = scan_dir(pred_dir, ext='.txt')
    list_pred = [os.path.basename(x) for x in list_pred]
    list_gt = scan_dir(gt_dir, ext='.txt')
    list_gt = [os.path.basename(x) for x in list_gt]

    fp_file = [item for item in list(list_pred) if item not in list_gt]
    frame_count += len(fp_file)
    for file in fp_file:
        file = os.path.join(pred_dir, file)
        rects = read_labels(file, skip)
        fp += len(rects)

    fn_file = [item for item in list(list_gt) if item not in list_pred]
    frame_count += len(fn_file)
    for file in fn_file:
        file = os.path.join(gt_dir, file)
        rects = read_labels(file, skip)
        fn += len(rects)
        object_count += len(rects)

    files = [item for item in list(list_pred) if item in list_gt]
    frame_count += len(files)

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
                if D[row, col] < th:
                    continue
                if row in usedRows or col in usedCols:
                    continue
                # print('find match', row, rects_pred[row], col, rects_gt[col])
                tp += 1

                usedRows.add(row)
                usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)  # unused prediction are false positive
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)  #  unused labels are false negative

        fp += len(unusedRows)
        fn += len(unusedCols)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('object count: {}'.format(object_count))
    print('tp: {}'.format(tp))
    print('fp: {}'.format(fp))
    print('fn: {}'.format(fn))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1: {}'.format(f1))
    print('iou tot / frame.... boh')

    return precision, recall, f1


def IoU_old(bboxa, bboxb):
    # print(bboxa,bboxb)

    box_a = [bboxa[0][0][0], bboxa[0][0][1], bboxa[0][1][0], bboxa[0][1][1]]
    box_b = [bboxb[0][0][0], bboxb[0][0][1], bboxb[0][1][0], bboxb[0][1][1]]

    p1 = box(minx=box_a[0], miny=box_a[1], maxx=box_a[2], maxy=box_a[3])
    p2 = box(minx=box_b[0], miny=box_b[1], maxx=box_b[2], maxy=box_b[3])

    intersection = p1.intersection(p2)

    iou = intersection.area / float(p1.area + p2.area - intersection.area)

    return iou


def evaluation_old(gt_dir, pred_dir):
    iou_tot = 0
    frame_count = 0
    fp = 0
    fn = 0
    tp = 0
    object_count = 0

    list_pred = sorted(os.listdir(pred_dir))
    list_gt = sorted(os.listdir(gt_dir))

    fp_file = [item for item in list(list_pred) if item not in list_gt]

    for file in fp_file:
        f_pred = open(pred_dir + file, 'r')
        n_object_pred, rect_pred = read_labels(f_pred)
        if n_object_pred != 0:
            fp = fp + n_object_pred

    for file in list_gt:
        name = os.path.join(gt_dir, file)
        f_gt = open(name, 'r')

        try:
            name = os.path.join(pred_dir, file)
            f_pred = open(name, 'r')

            n_object_gt, rect_gt = read_labels(f_gt)
            n_object_pred, rect_pred = read_labels(f_pred)

            gt_point = []
            if n_object_gt != 0:
                object_count += n_object_gt
                for i in range(n_object_gt):
                    c1 = rect_gt[0 + (2 * i)]
                    c2 = rect_gt[1 + (2 * i)]
                    gt_point.append((c1, c2))
                    # cv2.rectangle(frame, c1, c2, (0, 255, 0), 2)

            pred_point = []
            if n_object_pred != 0:

                for i in range(n_object_pred):
                    c1 = rect_pred[0 + (2 * i)]
                    c2 = rect_pred[1 + (2 * i)]
                    pred_point.append((c1, c2))
                    # cv2.rectangle(frame, c1, c2, (255, 0, 0), 2)

            if (len(gt_point) != len(pred_point)):
                # object_count += len(gt_point)

                if len(gt_point) > len(pred_point):
                    fn = fn + (len(gt_point) - len(pred_point))
                else:
                    fp = fp + (len(pred_point) - len(gt_point))

                if (len(gt_point) > 1):
                    if (len(gt_point) > 1):
                        iou_frame = 0
                        # calcolo iou per ogni oggetto
                        for i in range(len(gt_point)):
                            iou = []
                            gt = gt_point[i]
                            for j in range(len(pred_point)):
                                iou.append(IoU([gt], [pred_point[j]]))
                            # print('sono > 1')
                            iou_frame = iou_frame + max(iou)
                            if max(iou) >= 0.5:
                                tp += 1
                            else:
                                pass
                        iou_tot = iou_tot + (iou_frame / len(gt_point))
                else:
                    # object_count += len(gt_point)
                    iou_single = IoU(gt_point, pred_point)
                    if iou_single >= 0.5:
                        tp += 1
                    else:
                        pass

                    iou_tot = iou_tot + iou_single
            else:
                if (len(gt_point) > 1):
                    # object_count +=len(gt_point)
                    # aggiungo il numero di oggetti trovati
                    iou_frame = 0
                    # calcolo iou per ogni oggetto
                    for i in range(len(gt_point)):
                        iou = []
                        gt = gt_point[i]
                        for j in range(len(pred_point)):
                            iou.append(IoU([gt], [pred_point[j]]))
                        # print('sono > 1')
                        iou_frame = iou_frame + max(iou)
                        if max(iou) >= 0.5:
                            tp += 1
                        else:
                            pass
                    iou_tot = iou_tot + (iou_frame / len(gt_point))
                else:
                    # object_count +=len(gt_point)
                    # aggiungo il numero di oggetti trovati
                    iou_single = IoU(gt_point, pred_point)
                    if iou_single >= 0.5:
                        tp += 1
                    else:
                        pass

                    iou_tot = iou_tot + iou_single

            frame_count += 1
            # cv2.imshow('img', frame)
            # cv2.waitKey()
        except:
            iou_tot += 0
            frame_count += 1

    print('object count :' + str(object_count))
    print('tp :' + str(tp))
    print('fp :' + str(fp))
    print('fn :' + str(fn))
    print('precision :' + str(tp / (tp + fp)))
    print('recall :' + str(tp / (tp + fn)))
    print(iou_tot / frame_count)
