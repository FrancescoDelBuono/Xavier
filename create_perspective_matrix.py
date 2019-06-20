import os
import cv2
import argparse
import numpy as np
from tools.perspective import Points, create_matrix

"""
program to create the perspective matrix,
after the execution the user need to select the
4 points of the green garden and after that the matrix 
will be saved in the file './matrix.npy'
"""


# mouse callback function
def mouse_click(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_RBUTTONDOWN:
        points.remove()
        img = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        if points.add(x, y):
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)


parser = argparse.ArgumentParser(
    description='Run "create_perspective_matrix" to create the matrix for get the Top View')
parser.add_argument('--input',
                    required=True,
                    help='video to identify the point')

args = parser.parse_args()
input = args.input
# input = 'vid2.mp4'
if not os.path.isfile(input):
    print(input, 'is not a file')
    exit()

name = os.path.basename(input)
name = os.path.splitext(name)[0]
dir_name = os.path.dirname(input)

cap = cv2.VideoCapture(input)
r, frame = cap.read()

if not r:
    print('impossible to read the first frame')
    exit()

points = Points()

h, w, c = frame.shape
print(h, w, c)

text = '''left click to sign the point of the green garden'''
cv2.putText(frame, text, (3, 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
text = 'double right click to undo the sign'
cv2.putText(frame, text, (3, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
text = '''space to confirm the selection'''
cv2.putText(frame, text, (3, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

img = frame.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_click)

while 1:
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # esc
        print('esc pressed')
        break
    if key & 0xFF == 32:  # space
        print('space pressed')
        # if the number of selected points
        # is 4 then the matrix will be created
        if len(points.points) == 4:
            break

cv2.destroyAllWindows()

if len(points.points) == 4:
    print('save points')
    matrix = create_matrix(points.points, h, w)
    # outfile = './data/config/matrix.npy'
    outfile = './matrix.npy'

    np.save(outfile, matrix)
    matrix = np.load(outfile)
    # change frame perspective
    result = cv2.warpPerspective(frame.copy(), matrix, (640, 480), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # Visualize the perspective change
    while 1:
        cv2.imshow('frame', result)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # esc
            print('esc pressed')
            break
