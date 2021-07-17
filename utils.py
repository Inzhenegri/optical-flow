import cv2


W = 400
SIZE = (W, W, 3)
X_SHAPE = 700
Y_SHAPE = 540


def my_line(img, start, end):
    thickness = 2
    line_type = 8
    cv2.line(
        img=img,
        pt1=start,
        pt2=end,
        color=(255, 0, 0),
        thickness=thickness,
        lineType=line_type
    )


def my_line_red(img, start, end):
    thickness = 2
    line_type = 8
    cv2.line(
        img=img,
        pt1=start,
        pt2=end,
        color=(255, 255, 0),
        thickness=thickness,
        lineType=line_type
    )
