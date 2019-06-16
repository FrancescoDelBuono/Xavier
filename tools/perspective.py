import os
import cv2
import numpy as np

from scipy.spatial import distance as dist


#HEIGHT = 480
#WIDTH = 640

# video path for the first frame, center, color, frame, boolean True if we are at the first frame
def bird_eye_view(points, matrix, color, frame):
    c = np.array([points], dtype=np.float32)
    c = np.array([c])
    dest = cv2.perspectiveTransform(c, matrix)
    before = np.copy(frame)
    cv2.circle(frame, (int(dest[0][0][0]), int(dest[0][0][1])), 10, color, -1)
    return before, frame


class Points:
    def __init__(self):
        self.points = []

    def add(self, x, y):
        if len(self.points) < 4:
            self.points.append([x, y])
            return True
        return False

    def remove(self):
        self.points = []


def order_points_distance(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def order_points(pts):
    assert len(pts) == 4

    left = []
    right = []

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    # centroid = (, sum(y) / len(pts))
    x = sum(x) / len(pts)
    y = sum(y) / len(pts)

    for c in pts:
        if c[0] < x:
            left.append(c)
        else:
            right.append(c)

    if len(right) != len(left):
        raise ValueError('Impossible find corners')

    if left[0][1] < left[1][1]:
        tl = left[0]
        bl = left[1]
    else:
        tl = left[1]
        bl = left[0]

    if right[0][1] < right[1][1]:
        tr = right[0]
        br = right[1]
    else:
        tr = right[1]
        br = right[0]

    return [tl, tr, br, bl]


def create_matrix(points,height,width):
    print('matrix creation')
    view_h = height
    view_w = width
    garden_width = 200
    garden_height = garden_width / 1.25

    x = view_w / 2 - 100
    y = view_h / 2 - 125

    points = order_points(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[x, y], [x + garden_width, y], [x + garden_width, y + garden_height], [x, y + garden_height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return matrix
