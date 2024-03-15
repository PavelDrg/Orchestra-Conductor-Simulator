import cv2
from PIL import Image

from util import get_limits


color = [0, 255, 0]  # color in BGR colorspace
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    print(bbox)

    cv2.imshow('Conductor Cam', frame)

    if cv2.waitKey(1) & 0xFF == 27:    # 27 is the ASCII code for the escape key
        break

cap.release()

cv2.destroyAllWindows()

