import os
import cv2
import time
import argparse

from tools.utils import read_labels

"""
function to plot a ground truth on a video
to check how good is the ground truth

frame 0 label examples 
file_name: 000.txt
content:
2
100 100 150 150
200 200 210 230
"""

def main():
    parser = argparse.ArgumentParser(description='Run "timeseries_converter"')
    parser.add_argument('--input',
                        required=True,
                        help='file to detect and track')

    parser.add_argument('--label',
                        required=True,
                        help='directory where there are saved the label of the file')

    args = parser.parse_args()

    input = args.input
    if not os.path.isfile(input):
        print(input, 'is not a file')
        return

    label_dir = args.label
    if not os.path.isdir(label_dir):
        print('label_dir {} not found'.format(label_dir))
        return

    cap = cv2.VideoCapture(input)

    count = 0
    t1 = time.time()

    while True:
        t2 = time.time()
        if t2 - t1 < 0.10:
            continue
        t1 = t2

        r, frame = cap.read()
        if not r:
            break

        name = '{:03d}'.format(count)
        file_name = os.path.join(label_dir, name + '.txt')
        rects_gt = []

        if os.path.isfile(file_name):
            rects_gt = read_labels(file_name, skip=True)
        for rect in rects_gt:
            xA, yA, xB, yB = rect
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow("camera", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        count += 1

    cap.release()


if __name__ == '__main__':
    main()
