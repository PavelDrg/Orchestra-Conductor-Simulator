import cv2
from PIL import Image

from util import get_limits

color = [0, 255, 0]  # color in BGR colorspace
cap = cv2.VideoCapture(0)

num_slices = 8  # Number of vertical slices

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        screen_width = frame.shape[1]  # Width of the frame
        slice_width = screen_width // num_slices

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        slice_index = center_x // slice_width
        print("Object is in slice:", slice_index + 1)

    cv2.imshow('Conductor Cam', frame)

    if cv2.waitKey(1) & 0xFF == 27:    # 27 is the ASCII code for the escape key
        break

cap.release()
cv2.destroyAllWindows()
