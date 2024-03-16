import cv2
import numpy as np
from PIL import Image
import requests
import imutils
import threading
import os

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

            slice_index = center_x // slice_width
            print("Object is in slice:", slice_index + 1)

            # Determine the image path based on the slice index
            image_folder = "folder1"
            if 1 <= slice_index + 1 <= 2:
                image_name = "img1.png"
            elif 3 <= slice_index + 1 <= 4:
                image_name = "img2.png"
            elif 5 <= slice_index + 1 <= 6:
                image_name = "img3.png"
            elif 7 <= slice_index + 1 <= 8:
                image_name = "img4.png"
            else:
                # Default image
                image_name = "default.png"

            # Read logo and resize
            image_path = os.path.join(image_folder, image_name)
            logo = cv2.imread(image_path)
            size = 100
            logo = cv2.resize(logo, (size, size))

            # Create a mask of logo
            img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            # Region of Image (ROI), where we want to insert logo
            roi = img[-size-10:-10, -size-10:-10]

            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

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
