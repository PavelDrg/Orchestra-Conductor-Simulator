import cv2
import numpy as np
import imutils
import pygame
import threading
import os
from util import get_limits
import pyaudiowpatch as pyaudio
import time
import wave

color = [0, 255, 0]  # color in BGR colorspace

# Audio recording parameters
DURATION = 10.0
CHUNK_SIZE = 512
temp_filename = "D:/Proiecte AM/1/recordings/temp.wav"
loopback_filename = "D:/Proiecte AM/1/recordings/loopback_record.wav"

# Initialize pygame mixer
pygame.mixer.init()

# Flag to control detection process
start_detection = False

# Previously detected slice index
prev_slice_index = None

# Currently playing sound channel
current_channel = None

# Number of vertical slices
num_slices = 8

# Current folder indices for notes and images
current_notes_folder = 1
current_img_folder = 1


# Function to handle audio recording and playback concurrently
def record_audio():
    with pyaudio.PyAudio() as p:
        try:
            # Get default WASAPI info
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            return

        # Get default WASAPI speakers
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print(
                    "Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                return

        print(f"Recording from: ({default_speakers['index']}){default_speakers['name']}")

        # Start playback of existing recording if file exists
        if os.path.exists(loopback_filename):
            threading.Thread(target=play_loopback_audio, args=(p,)).start()

        # Prepare for recording
        wave_file = wave.open(temp_filename, 'wb')
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        with p.open(format=pyaudio.paInt16,
                    channels=default_speakers["maxInputChannels"],
                    rate=int(default_speakers["defaultSampleRate"]),
                    frames_per_buffer=CHUNK_SIZE,
                    input=True,
                    input_device_index=default_speakers["index"],
                    stream_callback=callback
                    ) as stream:
            """
            Open a PA stream via context manager.
            After leaving the context, everything will
            be correctly closed(Stream, PyAudio manager)            
            """
            print(f"The next {DURATION} seconds will be recorded and played back.")
            time.sleep(DURATION)  # Blocking execution while recording and playing back

        wave_file.close()
        # Add a delay to ensure recording is saved
        time.sleep(0.3)

        # Delete the old loopback_record.wav file if it exists
        if os.path.exists(loopback_filename):
            os.remove(loopback_filename)

        # Rename the temp file to loopback_record.wav after a delay
        os.rename(temp_filename, loopback_filename)


# Function to handle playback of loopback_record.wav
def play_loopback_audio(p):
    wave_file = wave.open(loopback_filename, 'rb')
    stream_out = p.open(format=p.get_format_from_width(wave_file.getsampwidth()),
                        channels=wave_file.getnchannels(),
                        rate=wave_file.getframerate(),
                        output=True)
    data = wave_file.readframes(CHUNK_SIZE)
    while data:
        stream_out.write(data)
        data = wave_file.readframes(CHUNK_SIZE)

    wave_file.close()
    stream_out.stop_stream()
    stream_out.close()



# Function to fetch images from the webcam
def fetch_images(camera_index):
    global start_detection, prev_slice_index, current_channel, current_notes_folder, current_img_folder
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = imutils.resize(img, width=800)  # Adjust the resize dimensions as needed

        # Add texts for start and stop
        start_text = "Start (s)    Stop (q)    Choir (1)    Brass(2)    Strings(3)"
        cv2.putText(img, start_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
                    sound_path = os.path.join(f"notes_{current_notes_folder}", f"{slice_index + 1}.mp3")
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
                image_folder = f"img_{current_img_folder}"
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
            # Stop the currently playing sound abruptly
            if current_channel is not None and current_channel.get_busy():
                current_channel.stop()
        elif key == ord('r'):  # Press 'r' to start audio recording
            threading.Thread(target=record_audio).start()
        elif key == ord('1'):  # Press '1' to switch to notes and images folder 1
            current_notes_folder = 1
            current_img_folder = 1
        elif key == ord('2'):  # Press '2' to switch to notes and images folder 2
            current_notes_folder = 2
            current_img_folder = 2
        elif key == ord('3'):  # Press '3' to switch to notes and images folder 3
            current_notes_folder = 3
            current_img_folder = 3
        elif key == ord('4'):  # Press '4' to switch to notes and images folder 4
            current_notes_folder = 4
            current_img_folder = 4
        elif key == ord('5'):  # Press '5' to switch to notes and images folder 5
            current_notes_folder = 5
            current_img_folder = 5
        elif key == ord('6'):  # Press '6' to switch to notes and images folder 6
            current_notes_folder = 6
            current_img_folder = 6
        elif key == ord('7'):  # Press '7' to switch to notes and images folder 7
            current_notes_folder = 7
            current_img_folder = 7


    # Release the capture
    cap.release()

# Create a thread for fetching images from the webcam (assuming camera index 1)
camera_index = 0
fetch_thread = threading.Thread(target=fetch_images, args=(camera_index,))
fetch_thread.start()

# Dummy loop to keep the main thread alive
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
        break

# Wait for the image fetching thread to finish
fetch_thread.join()
cv2.destroyAllWindows()
