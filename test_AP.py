import os
import cv2
import time
import warnings

warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from yolov3.detection_2 import Yolo
from tools.utils import non_max_suppression, read_labels
from tools.metrics import match_pred_gt

detector_types = ['hog', 'yolov3', 'yolov3Conf']
trackers_types = ['centroid', 'sort', 'open']


def main():
    parser = argparse.ArgumentParser(description='Run "timeseries_converter"')
    parser.add_argument('--input',
                        required=True,
                        help='file to detect and track')

    parser.add_argument('--label_dir',
                        required=True,
                        help='directory where there are saved the label of the file')

    parser.add_argument('--detector',
                        default='yolov3',
                        help='detector to use [yolov3, yolov3Conf, hog]')

    parser.add_argument('--conf',
                        default='yolov3',
                        help='configuration dir for yolov3Conf')

    parser.add_argument('--th',
                        default=0.5,
                        type=float,
                        help='number of frame to skip after detection')


    parser.add_argument('--show',
                        action='store_true',
                        help='if show the video in output')

    args = parser.parse_args()

    input = args.input
    if not os.path.isfile(input):
        print(input, 'is not a file')
        return

    name = os.path.basename(input)
    name = os.path.splitext(name)[0]

    detector_name = args.detector

    if detector_name == 'yolov3Conf':
        conf_dir = args.conf
        if not os.path.isdir(conf_dir):
            print('conf is not a directory')
            return
        conf = os.path.join(conf_dir, 'yolov3.cfg')
        weights = os.path.join(conf_dir, 'yolov3.pt')

        if not os.path.isfile(conf):
            print('{} doesn\'t exist'.format(conf))
            return

        if not os.path.isfile(weights):
            print('{} doesn\'t exist'.format(weights))
            return

    label_dir = args.label_dir
    if not os.path.isdir(label_dir):
        print('label_dir {} not found'.format(label_dir))
        return
    th = args.th
    if th < 0 or th > 1:
        raise argparse.ArgumentTypeError("%d must be between 0 and 1" % th)

    show_video = args.show

    print('input: {}'.format(input))
    print('label dir:', label_dir)
    print('detector: {}'.format(detector_name))
    print('show_video: {}'.format(show_video))

    detector = None
    if detector_name == 'hog':
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    elif detector_name == 'yolov3':
        detector = Yolo(weights='data/config/yolov3/yolov3.pt', cfg='data/config/yolov3/yolov3.cfg')
    elif detector_name == 'yolov3Conf':
        detector = Yolo(weights=weights, cfg=conf)
    else:
        print('Incorrect detector name')
        print('Available detectors are:')
        for d in detector_types:
            print(d)

    tracker = None

    cap = cv2.VideoCapture(input)

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    history = []

    y_true = []
    y_pred = []
    confidences = []
    overlaps = []



    while True:
        r, frame = cap.read()
        if not r:
            break

        rects = []
        if detector_name == 'hog':
            # frame = cv2.resize(frame, (640, 480))  # Downscale to improve frame rate
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image
            rects, weights = detector.detectMultiScale(gray_frame)

            # rects = np.array([[x, y, x + w, y + h] for i, (x, y, w, h) in enumerate(rects) if weights[i] > 0.7])
            # rects = non_max_suppression(rects, overlap_thresh=0.65)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            rects, picks = non_max_suppression(rects, overlap_thresh=0.65, index=True)
            weights = np.asarray(weights)
            weights = weights[picks]

            confidence = [w for w in weights]

        else:
            rects, confidence = detector.detect_image_confidence(frame)

        for rect in rects:
            xA, yA, xB, yB = rect
            history.append({
                'frame_id': count,
                'xA': xA,
                'yA': yA,
                'xB': xB,
                'yB': yB,
            })

            if show_video:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        if len(rects) > 0:
            name = '{:03d}'.format(count)
            file_name = os.path.join(label_dir, name + '.txt')
            rects_gt = []

            if os.path.isfile(file_name):
                rects_gt = read_labels(file_name, skip=True)

            overlapping = match_pred_gt(rects, rects_gt)
            overlaps += overlapping
            confidences += confidence

        if show_video:
            cv2.imshow("camera", frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        if count % 100 == 0 or count == total_frame:
            print('Processed {0:.1f}% of video.'.format(
                count * 100.0 / total_frame))
        count += 1

    y_true = [True if val > th else False for val in overlaps]
    y_scores = confidences
    ap = average_precision_score(y_true, y_scores)

    print('AP: ', ap)

    cap.release()


if __name__ == '__main__':
    main()
