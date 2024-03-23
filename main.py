import cv2
import numpy as np
import imutils
import pygame
import threading
import os
from util import get_limits

color = [0, 255, 0]  # color in BGR colorspace

# Initialize pygame mixer
pygame.mixer.init()


# Function to fetch images from the webcam
def fetch_images(camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = imutils.resize(img, width=800)  # Adjust the resize dimensions as needed

        # Perform object detection on the fetched frame
        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lowerLimit, upperLimit = get_limits(color)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        bbox = cv2.boundingRect(mask)

        if bbox[2] > 0 and bbox[3] > 0:
            x1, y1, w, h = bbox
            center_x = x1 + w // 2
            screen_width = img.shape[1]  # Width of the frame
            slice_width = screen_width // num_slices

            slice_index = center_x // slice_width
            print("Object is in slice:", slice_index + 1)

            # Play sound based on slice index
            sound_folder = "notes_piano"
            sound_name = f"{slice_index + 1}.mp3"
            sound_path = os.path.join(sound_folder, sound_name)
            if os.path.exists(sound_path):
                try:
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                except Exception as e:
                    print("Error playing sound:", e)

            # Determine the image path based on the slice index
            image_folder = "img_piano"
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

            # Overlay image on the frame
            img[-size - 10:-10, -size - 10:-10] = logo

        cv2.imshow('Conductor Cam', img)

        if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
            break

    # Release the capture
    cap.release()


# Create a thread for fetching images from the webcam (assuming camera index 1)
camera_index = 1
fetch_thread = threading.Thread(target=fetch_images, args=(camera_index,))
fetch_thread.start()

num_slices = 8  # Number of vertical slices

# Dummy loop to keep the main thread alive
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
        break

# Wait for the image fetching thread to finish
fetch_thread.join()
cv2.destroyAllWindows()