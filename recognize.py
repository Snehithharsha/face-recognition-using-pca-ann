import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")

label_map = {}

with open("labels.txt", "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        label_map[int(idx)] = name

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 50:
            name = label_map[label]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y),
                      (x+w, y+h),
                      (0, 255, 0), 2)

        cv2.putText(frame,
                    name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()