import cv2
from vidgear.gears import PiGear
from vidgear.gears import NetGear
from vidgear.gears import VideoGear

options_netgear = {
    'flag': 0,
    'copy': False,
    'track': False,
    "jpeg_compression_quality": 80,
    'jpeg_compression_fastupsample': True
}

stream = cv2.VideoCapture(0)

server = NetGear(
    # address='192.168.11.145', # school network
    address='192.168.11.137', # home network
    port='5454',
    protocol='tcp',
    pattern=2,
    logging=True,
    receive_mode=False,
    **options_netgear
)

while True:
    try:
        _, frame = stream.read()

        if frame is None:
            break

        server.send(frame=frame)
        print('sending...')
    except KeyboardInterrupt:
        break

stream.release()
server.close()
