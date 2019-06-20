import os
import cv2
from tools.perspective import Points, create_matrix


# mouse callback function
def mouse_click(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_RBUTTONDOWN:
        points.remove()
        img = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        if points.add(x, y):
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)


def create_birdeye(input):
    # function to compute perspective matrix online
    # given a video to get the top view
    input = input
    if not os.path.isfile(input):
        print(input, 'is not a file')
        exit()

    global frame
    cap = cv2.VideoCapture(input)
    r, frame = cap.read()

    if not r:
        print('impossible to read the first frame')
        exit()

    global points
    points = Points()

    h, w, c = frame.shape

    text = '''left click to sign the point of the green garden'''
    cv2.putText(frame, text, (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text = '''in order TOP-LEFT, BOTTOM-LEFT, BOTTOM-RIGHT, TOP-RIGHT '''
    cv2.putText(frame, text, (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text = 'double right click to undo the sign'
    cv2.putText(frame, text, (3, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text = '''space to confirm the selection'''
    cv2.putText(frame, text, (3, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    global img
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
            if len(points.points) == 4:
                break

    cv2.destroyAllWindows()

    if len(points.points) == 4:
        if len(points.points) == 4:
            print('save points')
            matrix = create_matrix(points.points,h,w)
            return matrix

    return None
