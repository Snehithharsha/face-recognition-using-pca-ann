import cv2

cap = cv2.VideoCapture("http://192.168.1.10:8080/video")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Mobile Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()