import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fuzzy_logic import fuzzy_emotion

model = load_model("model/emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_labels = ['angry','happy','sad','surprise','neutral','fear','disgust']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = face.reshape(1,48,48,1)

        prediction = model.predict(face)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction)
        fuzzy_result = fuzzy_emotion(confidence)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,
                    f"{emotion} ({fuzzy_result})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()