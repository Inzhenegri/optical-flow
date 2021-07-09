from vidgear.gears import NetGear
import cv2
import numpy as np
import matplotlib.pyplot as plt


options = {
    'flag': 0,
    'copy': False,
    'track': False
}

client = NetGear(
    # address='192.168.11.145', # school network
    address='192.168.11.137', # home network
    port='5555',
    pattern=2,
    receive_mode=True,
    logging=True,
    protocol='tcp',
    **options
)

frame1 = client.recv()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(a=frame1)
hsv[..., 1] = 255 # 255 stands for green color


def start():
    global prvs
    while True:
        frame2 = client.recv()

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

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(src=mag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', bgr)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        prvs = next_frame


start()

cv2.destroyAllWindows()
client.close()
