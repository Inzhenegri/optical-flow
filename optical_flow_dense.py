import time

from vidgear.gears import NetGear
import cv2
import numpy as np

from utils import (
    W,
    SIZE,
    X_SHAPE,
    Y_SHAPE,
    my_line,
    my_line_red
)


options = {
    'flag': 0,
    'copy': False,
    'track': False
}

client = NetGear(
    address='192.168.11.145', # school network
    # address='192.168.11.137', # home network
    port='5555',
    pattern=2,
    receive_mode=True,
    logging=True,
    protocol='tcp',
    **options
)


x = []
y = []
t = 0
delta_time = 0

frame1 = client.recv()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

rook_image = np.zeros(SIZE, dtype=np.uint8)
rook_window = 'Drawing 2: Rook'

hsv = np.zeros_like(a=frame1)
hsv[..., 1] = 255 # 255 stands for green color

print(''' Simple Linear Blender
-----------------------
* Enter alpha [0.0-1.0]: ''')

alpha = None


def start():
    global prvs, rook_image, delta_time, alpha

    # input_alpha = float(input('your value: ').strip())
    input_alpha = 0.5

    i = 0

    while True:
        start_time = time.time()
        frame2 = client.recv()

        # alpha value
        if 0 <= input_alpha <= 1:
            alpha = input_alpha 

        if frame2 is None:
            break

        next_frame = cv2.cvtColor(src=frame2, code=cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev=prvs,
            next=next_frame,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        dvx = -np.ma.average(a=flow[..., 0])
        dvy = np.ma.average(a=flow[..., 1])

        rook_image = cv2.resize(src=rook_image, dsize=(X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(src=frame2, dsize=(X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_AREA)

        my_line(img=rook_image, start=(X_SHAPE - 50, Y_SHAPE - 50), end=(X_SHAPE - 50 + int((500 * dvx) // 10), Y_SHAPE - 50 + int((500 * dvy) // 10)))
        my_line(img=rook_image, start=(X_SHAPE - 50, Y_SHAPE - 50), end=(X_SHAPE - 50, Y_SHAPE - 50 + int((500 * dvy) // 10)))
        my_line_red(img=rook_image, start=(X_SHAPE - 50, Y_SHAPE - 50), end=(X_SHAPE - 50 + int((500 * dvx) // 10), Y_SHAPE - 50))

        # add text
        cv2.putText(img=rook_image, text='X', org=(X_SHAPE - 50, Y_SHAPE - 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
        cv2.putText(img=rook_image, text='Y', org=(X_SHAPE - 60, Y_SHAPE - 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))


        if rook_image is None:
            raise Exception('rook_image is None.')
        if next_frame is None:
            raise Exception('next_frame is None.')

        # blend images
        beta = (1.0 - alpha)

        dst = cv2.addWeighted(src1=frame2, alpha=alpha, src2=rook_image, beta=beta, gamma=0.0, dst=rook_image)
        cv2.imshow('dst', dst)


        if t > 10:
            break

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(src=mag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)

        # cv2.imshow('bgr', bgr)
        # cv2.imshow('frame', frame2)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        rook_image = np.zeros(SIZE, dtype=np.uint8)
        prvs = next_frame


start()

cv2.destroyAllWindows()
client.close()
