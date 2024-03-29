import os
import glob
import numpy as np


def non_max_suppression(boxes, overlap_thresh, index=False):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        if index:
            return [], []
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    if index:
        return boxes[pick].astype("int"), pick

    return boxes[pick].astype("int")


def scan_dir(img_dir, ext=''):
    # scan a directory to get all file that end with ext
    files = [entry for entry in glob.iglob("{}/**".format(img_dir), recursive=True) if
             os.path.isfile(entry) and entry.endswith(ext)]
    return files


def read_labels(file, skip=False, with_confidence=False):
    if not os.path.isfile(file):
        raise ValueError('{} is not a file'.format(file))

    with open(file, 'r') as f:
        lines = f.readlines()

    if len(lines) > 0 and skip:
        lines.pop(0)

    rects = []
    confidence = []
    for line in lines:
        s = line.split()
        rect = [int(x) for x in s]
        if with_confidence:
            c = rect.pop(0)
            confidence.append(c)
        if len(rect) != 4:
            raise ValueError('no rect found')

        rects.append(rect)
    if with_confidence:
        return rects, confidence
    return rects
