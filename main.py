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

# Flag to control detection process
start_detection = False

# Previously detected slice index
prev_slice_index = None

# Currently playing sound channel
current_channel = None

# Function to fetch images from the webcam
def fetch_images(camera_index):
    global start_detection, prev_slice_index, current_channel
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = imutils.resize(img, width=800)  # Adjust the resize dimensions as needed

        # Add texts for start and stop
        start_text = "Start (s)"
        stop_text = "Stop (q)"
        cv2.putText(img, start_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Calculate the width of start text to place stop text next to it
        start_text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        stop_text_size = cv2.getTextSize(stop_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        # Add stop text to the right of start text
        stop_text_x = 40 + start_text_size[0] + 20  # Adjust spacing as needed
        cv2.putText(img, stop_text, (stop_text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if start_detection:
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

                # Play sound based on slice index if the slice has changed
                if slice_index != prev_slice_index:
                    sound_path = os.path.join("notes_piano", f"{slice_index + 1}.mp3")
                    if os.path.exists(sound_path):
                        try:
                            # Fade out the previous sound if playing
                            if current_channel is not None and current_channel.get_busy():
                                current_channel.fadeout(500)  # Fade out over 500 milliseconds
                            sound = pygame.mixer.Sound(sound_path)
                            current_channel = sound.play(fade_ms=500)  # Fade in over 500 milliseconds
                            prev_slice_index = slice_index
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

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('s'):  # Press 's' to start detection
            start_detection = True
        elif key == ord('q'):  # Press 'q' to stop detection
            start_detection = False

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