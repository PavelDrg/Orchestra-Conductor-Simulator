import cv2
from PIL import Image
import numpy as np
import requests
import imutils
import threading

from util import get_limits

color = [0, 255, 0]  # color in BGR colorspace

# Function to fetch images from the URL
def fetch_images(url):
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)  # Adjust the resize dimensions as needed

        # Perform object detection on the fetched frame
        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lowerLimit, upperLimit = get_limits(color)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            screen_width = img.shape[1]  # Width of the frame
            slice_width = screen_width // num_slices

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 0)

            slice_index = center_x // slice_width
            print("Object is in slice:", slice_index + 1)

        cv2.imshow('Conductor Cam', img)

        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
            break

# Replace the below URL with your own. Make sure to add "/shot.jpg" at the end.
url = "http://192.168.100.32:8080/shot.jpg"

# Create a thread for fetching images
fetch_thread = threading.Thread(target=fetch_images, args=(url,))
fetch_thread.start()

num_slices = 8  # Number of vertical slices

# Dummy loop to keep the main thread alive
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
        break

# Wait for the image fetching thread to finish
fetch_thread.join()
cv2.destroyAllWindows()
