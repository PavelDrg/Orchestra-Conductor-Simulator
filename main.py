import cv2


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    cv2.imshow('Conductor Cam', frame)

    if cv2.waitKey(1) & 0xFF == 27:    # 27 is the ASCII code for the escape key
        break

cap.release()

cv2.destroyAllWindows()

