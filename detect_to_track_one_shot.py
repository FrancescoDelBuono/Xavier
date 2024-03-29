import os
import cv2
import time
import warnings

import numpy as np
import pandas as pd

from trackers.Tracker import TrackableObject, Sort, CentroidTracker, OpenTracker
from tools.utils import non_max_suppression
from tools.create_matrix import create_birdeye

from yolov3.detection_2 import Yolo

warnings.filterwarnings("ignore")

detector_types = ['hog', 'yolov3']
trackers_types = ['centroid', 'sort', 'open']

"""
Xavier algorithm to detect and track people
2 detector:
    - hog
    - yolov3
3 trackers:
    - OpenTracker
    - Centroid
    - Sort
with the possibility to change the perspective of the image
and visualize also the trace of the people detected 
"""


def main():
    # tracker_name = 'centroid'
    # tracker_name = 'sort'
    tracker_name = 'open'

    detector_name = 'yolov3'
    # detector_name = 'hog'

    save_video = True   # if save the video with detected and tracked objects
    save_label = True   # if save the label of the detected and tracked objects

    show_video = True   # if show the video
    show_trace = True   # if show the trace of human on hte ground

    top_view = True     # if show and save the camera view from above

    input = "data/video/vid2.mp4"
    # input = "data/video/TownCentreXVID.avi"

    skip = 1
    if skip < 1:
        raise ValueError("%d is an invalid positive int value" % skip)

    print('input: ', input)
    print('detector: ', detector_name)
    print('tracker: ', tracker_name)
    print('skip: ', skip)
    print('save video: ', save_video)
    print('save label: ', save_label)
    print('show video: ', show_video)
    print('show trace: ', show_trace)
    print('top view: ', top_view)

    if not os.path.isfile(input):
        print(input, 'is not a file')
        return

    name = os.path.basename(input)
    name = os.path.splitext(name)[0]
    dir_name = os.path.dirname(input)

    # check top view matrix
    if top_view:
        matrix_file = 'data/config/matrix.npy'
        matrix = np.load(matrix_file)

    # create the matrix to
    # visualize the top view
    if top_view and False:
        matrix = create_birdeye(input)
        if matrix is None:
            print('impossible to create top view')
            return

    # instance the detector (hog or yolov3)
    detector = None
    if detector_name == 'hog':
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif detector_name == 'yolov3':
        # check the existence of configuration
        # and weights file for yolov3
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

    # only with OpenTracker we are able to predict
    # and to avoid the detection to each frame
    if tracker_name != 'open' and skip > 1:
        print('impossible skip frame without OpenTracker')
        return

    # instance the tracker (open, sort, or centroid)
    tracker = None
    if tracker_name == 'sort':
        tracker = Sort()

    elif tracker_name == 'centroid':
        tracker = CentroidTracker(maxDisappeared=10)

    elif tracker_name == 'open':
        if skip > 10:
            disap = int(skip * 1.5)
            tracker = OpenTracker(tracker='csrt', reinit=True, max_disappeared=disap, th=0.5, show_ghost=skip)
        else:
            tracker = OpenTracker(tracker='csrt', reinit=True, max_disappeared=20, th=0.5, show_ghost=10)

    else:
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackers_types:
            print(t)

    trackable_objects = {}

    history = []

    # cap = cv2.VideoCapture("data/video/drop.avi")
    cap = cv2.VideoCapture(input)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if save_video:
        # get the input video metadata
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = os.path.join(dir_name, 'result_' + name + '.avi')
        out = cv2.VideoWriter(output_file, fourcc, frame_fps, (frame_width, frame_height))
        # out = cv2.VideoWriter(output_file, fourcc, frame_fps, (640, 480))

    # background = np.zeros((HEIGHT, WIDTH, 3))

    count = 0
    while True:
        r, frame = cap.read()
        if not r:
            break

        h, w, c = frame.shape

        if top_view and count == 0:
            # using the first frame as background to change the perspective
            background = frame.copy()
            background = cv2.resize(background, (w, h))
            background = cv2.warpPerspective(background, matrix, (w, h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Detection
        rects = []  # empty rects to avoid previous prediction propagation when skip > 1
        if count % skip == 0:
            if detector_name == 'yolov3':
                rects = detector.detect_image(frame)
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image
                rects, weights = detector.detectMultiScale(gray_frame)  # Detection
                # keep only the detections with confidence greater than 0.7
                rects = np.array([[x, y, x + w, y + h] for i, (x, y, w, h) in enumerate(rects) if weights[i] > 0.7])
                # remove overlapped detections (no max suppression)
                rects = non_max_suppression(rects, overlap_thresh=0.65)

        # Tracking
        if tracker_name == 'open':
            objects = tracker.update(frame, rects)
        else:
            objects = tracker.update(rects)

        # Visualization
        for obj in objects:
            xA, yA, xB, yB, objectId = obj.astype(np.int)

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
            for obj in objects:
                xA, yA, xB, yB, objectId = obj.astype(np.int)
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

        if save_video:
            out.write(frame)

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
