import cv2
import numpy as np
import imutils
import pygame
import threading
import os
import pyaudiowpatch as pyaudio
import time
import wave

from util import get_limits

color = [0, 255, 0]  # culoarea "baghetei de dirijor" in BGR (blue green red)

# Parametrii pentru inregistrarea audi (loop-ul)
DURATION = 10.0
CHUNK_SIZE = 512
temp_filename = "D:/Proiecte AM/1/recordings/temp.wav"
loopback_filename = "D:/Proiecte AM/1/recordings/loopback_record.wav"

# Initializare pygame mixer
pygame.mixer.init()

# Flag pentru procesul de detectie
start_detection = False

# Ultimul slice detectat
prev_slice_index = None

# Canalul de sunet curent
current_channel = None

# Numarul de slice-uri in care impartim ecranul (8 slice-uri pentru 8 note)
num_slices = 8

# Indicii curenti pentru folderele de imagini si sunete
current_notes_folder = 1
current_img_folder = 1

# Flag pentro looping
looping_enabled = False


# Functia care ia o "marja de eroare" in jurul culorii selectate pentru detectie
def get_limits(color):
    c = np.uint8([[color]])  # convertim valoarea BGR in HSV (hue saturation volume)
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


# Functie pentru inregistrarea loop-ului si playback simultan
def record_audio():
    with pyaudio.PyAudio() as p:
        try:
            # Info default WASAPI (Windows Audio Session API)
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Sistemul nu are WASAPI. Iesire...")
            return

        # Selectare boxe WASAPI default
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Se incearca gasirea device-ului de loopback cu acelasi nume si Loopback suffix
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print(
                    "Device-ul default de loopback nu a fost gasit.\n\nIesire...\n")
                return

        print(f"Se inregistreaza din: ({default_speakers['index']}){default_speakers['name']}")

        # Se incepe playback cu ultima inregistrare (daca aceasta exista).
        if os.path.exists(loopback_filename):
            threading.Thread(target=play_loopback_audio, args=(p,)).start()

        # Pregatire pentru recording
        wave_file = wave.open(temp_filename, 'wb')
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
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
            Se deschide un stream PA via context manager.
            Dupa iesirea din context, se opreste totul(Stream, PyAudio manager).        
            """
            print(f"Urmatoarele {DURATION} secunde vor fi inregistrate.")
            time.sleep(DURATION)  # Se blocheaza executia in timpul inregistrarii si playback-ului.

        wave_file.close()
        print("Inregistrare incheiata.")
        # Am adaugat un delay sa fiu sigur ca totul se executa ok
        time.sleep(0.3)

        # Stergem fisierul vechi loopback_record.wav daca exista
        if os.path.exists(loopback_filename):
            os.remove(loopback_filename)

        # Redenumim fisierul temporar (acum el devine inregistrarea de baza)
        os.rename(temp_filename, loopback_filename)


# Functie pentru playback-ul fisierului loopback_record.wav
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


# Functia principala pentru webcam si controlul pe butoane
def fetch_images(camera_index):
    global start_detection, prev_slice_index, current_channel, current_notes_folder, current_img_folder, looping_enabled
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = imutils.resize(img, width=800)

        # In caz ca vrem sa vedem segmentele (se pot comenta urmatoarele 3 linii daca nu)
        segment_width = img.shape[1] // num_slices
        for i in range(1, num_slices):
            cv2.line(img, (segment_width * i, 0), (segment_width * i, img.shape[0]), (0, 0, 255), 2)

        # Text pentru controale
        start_text = ""
        cv2.putText(img, start_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if start_detection:
            # Se face detectia pentru frame-ul obtinut
            hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lowerLimit, upperLimit = get_limits(color)

            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

            bbox = cv2.boundingRect(mask)

            if bbox[2] > 0 and bbox[3] > 0:
                x1, y1, w, h = bbox
                center_x = x1 + w // 2
                screen_width = img.shape[1]  # Latimea frame-ului
                slice_width = screen_width // num_slices

                slice_index = center_x // slice_width
                print("Obiectul se afla in slice-ul:", slice_index + 1)

                # Activeaza sunetul pe baza slice-ului curent daca s-a schimbat.
                if slice_index != prev_slice_index:
                    sound_path = os.path.join(f"notes_{current_notes_folder}", f"{slice_index + 1}.mp3") # A ajutat mult denumirea folderelor si a notelor intr-o maniera similara
                    if os.path.exists(sound_path):
                        try:
                            # Fade out pentru ultimul sunet activat.
                            if current_channel is not None and current_channel.get_busy():
                                current_channel.fadeout(500)  # Putem regla durata de fade out in functie de cat de brusca dorim trecerea
                            sound = pygame.mixer.Sound(sound_path)
                            current_channel = sound.play(fade_ms=500)  # La fel si pentru fade in
                            prev_slice_index = slice_index
                        except Exception as e:
                            print("Eroare activare sunet:", e)

                # Selectam imaginea potrivita in functie de slice-ul detectat
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
                    # Imagine default (nu e nevoie)
                    image_name = "default.png"

                # Citire si resize pentru imagine
                image_path = os.path.join(image_folder, image_name)
                logo = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Incarcare imagine cu alpha channel (e nevoie pentru pastrarea transparentei)
                size = 150
                logo = cv2.resize(logo, (size, size))

                # Punem imaginile peste interfata noastra
                overlay_alpha(img, logo, (-size - 10, -size - 10))

        cv2.imshow('Simulator de dirijor :D', img)

        key = cv2.waitKey(1)
        if key == 27:  # Esc - inchiderea aplicatiei
            break
        elif key == ord('s'):  # s - start detectie
            start_detection = True
        elif key == ord('q'):  # q - stop detectie
            start_detection = False
            # Oprire brusca a sunetului activ
            if current_channel is not None and current_channel.get_busy():
                current_channel.stop()
        elif key == ord('r'):  # r - start iregistrare
            threading.Thread(target=record_audio).start()
        elif key == ord('1'):  # 1 - cor
            current_notes_folder = 1
            current_img_folder = 1
        elif key == ord('2'):  # 2 - trompeta
            current_notes_folder = 2
            current_img_folder = 2
        elif key == ord('3'):  # 3 - vioara
            current_notes_folder = 3
            current_img_folder = 3
        elif key == ord('4'):  # 4 - clopotei
            current_notes_folder = 4
            current_img_folder = 4
        elif key == ord('5'):  # 5 - pian
            current_notes_folder = 5
            current_img_folder = 5
        elif key == ord('6'):  # 6 - cor secundar
            current_notes_folder = 6
            current_img_folder = 6
        elif key == ord('7'):  # 7 - pad / un fel de orga
            current_notes_folder = 7
            current_img_folder = 7
        elif key == ord('p'):  # p - reset inregistrare
            if os.path.exists(loopback_filename):
                os.remove(loopback_filename)
                print("Inregistrare stearsa.")
        elif key == ord('l'):  # l - oprire/pornire loop
            looping_enabled = not looping_enabled
            if looping_enabled:
                # Verificacm daca sunetul nu e deja activ
                if current_channel is None or not current_channel.get_busy():
                    sound_path = "D:/Proiecte AM/1/recordings/loopback_record.wav"
                    if os.path.exists(sound_path):
                        try:
                            sound = pygame.mixer.Sound(sound_path)
                            # Punem sunetul in loop
                            current_channel = sound.play(loops=-1)
                        except Exception as e:
                            print("Error playing sound:", e)
            else:
                # Oprim loop-ul daca e pornit
                if current_channel is not None and current_channel.get_busy():
                    current_channel.stop()

    # Oprim captura
    cap.release()


# Functie pentru plasarea imaginilor tinand cont de transparenta
def overlay_alpha(frame, overlay, position):
    y, x = position

    # Verificam daca imaginea are un canal alpha
    if overlay.shape[2] == 4:  # Daca are 4 canale (incluzand canalul alpha)
        # Taiem overlay-ul din frame
        roi = frame[y:y + overlay.shape[0], x:x + overlay.shape[1]]

        # Extragem canalul alpha
        alpha = overlay[:, :, 3] / 255.0

        # Calculam inversul canalului
        inv_alpha = 1.0 - alpha

        # Le combinam folosind canalul alpha
        for c in range(0, 3):
            frame[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = (alpha * overlay[:, :, c] +
                                                                        inv_alpha * roi[:, :, c])
    else:  # Daca nu are canal alpha
        # Pur si simplu facem overlay direct cu imaginea nemodificata
        frame[y:y + overlay.shape[0], x:x + overlay.shape[1]] = overlay


# Cream un thread pentru obtinerea imaginilor de la camera
camera_index = 0 # Putem selecta diverse camere in functie de indexul lor (pentru mine telefonul are indexul 0 si webcam-ul de la laptop indexul 1
fetch_thread = threading.Thread(target=fetch_images, args=(camera_index,))
fetch_thread.start()

# Loop "de forma" ca sa tinem thread-ul principal activ
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc key to exit
        break

# Asteptam sa se termine obtinerea de imagini
fetch_thread.join()
cv2.destroyAllWindows()
