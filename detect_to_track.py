import os
import cv2
import time
import warnings

# warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd

from trackers.Tracker import TrackableObject, Sort, CentroidTracker, OpenTracker
from tools.utils import non_max_suppression
#from tools.perspective import HEIGHT, WIDTH
from tools.create_matrix import create_birdeye

from yolov3.detection_2 import Yolo

detector_types = ['hog', 'yolov3']
trackers_types = ['centroid', 'sort', 'open']


def main():
    parser = argparse.ArgumentParser(description='Run "timeseries_converter"')
    parser.add_argument('--input',
                        required=True,
                        help='file to detect and track')

    parser.add_argument('--detector',
                        default='yolov3',
                        help='detector to use [yolov3, hog]')

    parser.add_argument('--tracker',
                        default='open',
                        help='tracker to use [open, sort, centroid]')

    # parser.add_argument('--show',
    #                     action='store_true',
    #                     help='if show the video in output')

    parser.add_argument('--save',
                        action='store_true',
                        help='if save the final output')

    parser.add_argument('--label',
                        action='store_true',
                        help='if save the label')

    parser.add_argument('--trace',
                        action='store_true',
                        help='if show the trace')

    parser.add_argument('--top',
                        action='store_true',
                        help='if show the view from above')

    parser.add_argument('--matrix',
                        help='file contain matrix to change camera view')

    args = parser.parse_args()
    input = args.input
    tracker_name = args.tracker
    detector_name = args.detector

    save_video = args.save
    save_label = args.label

    # show_video = args.show
    show_video = True
    show_trace = args.trace

    top_view = args.top
    matrix_file = args.matrix

    if top_view:
        if matrix_file:
            if not os.path.isfile(matrix_file):
                print(matrix_file, 'is not a file')
                return

            matrix = np.load(matrix_file)
        else:
            matrix = create_birdeye(input)
            if matrix is None:
                print('impossible to create top view')
                return

    print('input: {}'.format(input))
    print('detector: {}'.format(detector_name))
    print('tracker: {}'.format(tracker_name))
    print('save_video: {}'.format(save_video))
    print('save_label: {}'.format(save_label))
    print('top_view: {}'.format(top_view))
    print('show_video: {}'.format(show_video))
    print('show_trace: {}'.format(show_trace))
    print('top view: ', top_view)

    if not os.path.isfile(input):
        print(input, 'is not a file')
        return

    name = os.path.basename(input)
    name = os.path.splitext(name)[0]
    dir_name = os.path.dirname(input)

    detector = None
    if detector_name == 'hog':
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    elif detector_name == 'yolov3':
        if not os.path.isfile('data/config/yolov3/yolov3.pt'):
            print('data/config/yolov3/yolov3.pt not found')
            return
        if not os.path.isfile('data/config/yolov3/yolov3.cfg'):
            print('data/config/yolov3/yolov3.cfg not found')
            return
        detector = Yolo(weights='data/config/yolov3/yolov3.pt', cfg='data/config/yolov3/yolov3.cfg')
    else:
        print('Incorrect detector name')
        print('Available detectors are:')
        for d in detector_types:
            print(d)

    tracker = None
    if tracker_name == 'sort':
        tracker = Sort()
    elif tracker_name == 'centroid':
        tracker = CentroidTracker(maxDisappeared=10)
    elif tracker_name == 'open':
        tracker = OpenTracker(tracker='csrt', reinit=True, max_disappeared=20, th=0.5, show_ghost=10)
    else:
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackers_types:
            print(t)

    trackable_objects = {}
    skip_frame = 10

    history = []

    # cap = cv2.VideoCapture("data/video/drop.avi")
    cap = cv2.VideoCapture(input)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if save_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = os.path.join(dir_name, 'result_' + name + '.avi')
        out = cv2.VideoWriter(output_file, fourcc, frame_fps, (frame_width, frame_height))
        # out = cv2.VideoWriter(output_file, fourcc, frame_fps, (640, 480))

    #background = np.zeros((h, w, 3))

    count = 0
    while True:
        # if count == 10:
        #     break

        r, frame = cap.read()
        if not r:
            break

        h,w,c = frame.shape

        if top_view and count == 0:
            background = frame.copy()
            background = cv2.resize(background, (w, h))
            background = cv2.warpPerspective(background, matrix, (w, h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Detection
        if detector_name == 'yolov3':
            rects = detector.detect_image(frame)
        else:
            # frame = cv2.resize(frame, (640, 480))  # Downscale to improve frame rate
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image
            rects, weights = detector.detectMultiScale(gray_frame)
            rects = np.array([[x, y, x + w, y + h] for i, (x, y, w, h) in enumerate(rects) if weights[i] > 0.7])
            rects = non_max_suppression(rects, overlap_thresh=0.65)

        # Tracking
        objects = None
        if tracker_name == 'open':
            objects = tracker.update(frame, rects)
        else:
            objects = tracker.update(rects)

        # Visualization
        for object in objects:
            xA, yA, xB, yB, objectId = object.astype(np.int)

            history.append({
                'frame_id': count,
                'object_id': objectId,
                'xA': xA,
                'yA': yA,
                'xB': xB,
                'yB': yB,
            })

            cX = int((xA + xB) / 2.0)
            cY = int((yA + yB) / 2.0)

            if objectId not in trackable_objects:
                trackable_objects[objectId] = TrackableObject(objectId, [xA, yA, xB, yB])
            else:
                trackable_objects[objectId].update([xA, yA, xB, yB])

            color = trackable_objects[objectId].color
            cv2.rectangle(frame, (xA, yA), (xB, yB), color, 2)

            text = "ID {}".format(objectId)
            cv2.putText(frame, text, (cX - 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.circle(frame, (cX, cY), 4, color, -1)

            if show_trace:
                for xA, yA, xB, yB in trackable_objects[objectId].rects[:]:
                    cX = int((xA + xB) / 2.0)
                    cv2.circle(frame, (cX, yB), 2, color, -1)

        # View From Above
        if top_view:
            top_image = background.copy()
            for object in objects:
                xA, yA, xB, yB, objectId = object.astype(np.int)
                color = trackable_objects[objectId].color

                cX = int((xA + xB) / 2.0)
                point = (cX, yB)

                point = np.array([point], dtype=np.float32)
                point = np.array([point])

                point = cv2.perspectiveTransform(point, matrix)

                point = tuple(np.squeeze(point))

                cv2.circle(top_image, point, 7, color, -1)

            cv2.imshow("top view", top_image)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Management
        if show_video:
            cv2.imshow("camera", frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # pass

        if save_video:
            out.write(frame)
            # pass

        # plt.imshow(frame[:, :, ::-1])
        # plt.grid(False)
        # plt.show()

        if count % 100 == 0 or count == total_frame:
            print('Processed {0:.1f}% of video.'.format(
                count * 100.0 / total_frame))
        count += 1

    cap.release()

    if save_video:
        out.release()

    if save_label:
        history = pd.DataFrame(history)
        output_file = os.path.join(dir_name, 'label_' + name + '.csv')
        history.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
