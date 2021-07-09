from vidgear.gears import NetGear
import cv2
import asyncio
import numpy as np


options = {
    'flag': 0,
    'copy': False,
    'track': False
}

client = NetGear(
    address='192.168.11.145', # school network
    # address='192.168.11.137', # home network
    port='5454',
    pattern=2,
    receive_mode=True,
    logging=True,
    protocol='tcp',
    **options
)

feature_params=  dict(
    maxCorners=100,
    qualityLevel=0.03,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

color = np.random.randint(low=0, high=255, size=(100, 3))

old_frame = client.recv()
old_gray = cv2.cvtColor(src=old_frame, code=cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(image=old_gray, mask=None, **feature_params)

# create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


async def start():
    global old_gray, p0, mask
    while True:
        frame = client.recv()

        if frame is None:
            break

        frame_gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=p0, nextPts=None, **lk_params)

        # select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), thickness=2)
                frame = cv2.circle(frame, (int(a), int(b)), radius=5, color=color[i].tolist(), thickness=-1)
        img = cv2.add(src1=frame, src2=mask)

        cv2.imshow('frame', img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        await asyncio.sleep(0.00001)


asyncio.run(main=start())

cv2.destroyAllWindows()
client.close()
