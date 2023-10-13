import cv2
import os
from keras.models import load_model
from twilio.rest import Client  # Import Twilio library for sending SMS
from pygame import mixer
import numpy as np
import geocoder
#Your Twilio credentials
account_sid = ''
twilio_phone_number = ''
emergency_contacts = ['']

mixer.init()
sound = mixer.Sound('alarm.wav')

location = geocoder.ip('me')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

sms_sent = False
location = None

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    
    location = geocoder.ip('me')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if np.all(rpred) == 1:
            lbl = 'Open'
        if np.all(rpred) == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if np.all(lpred) == 1:
            lbl = 'Open'
        if np.all(lpred) == 0:
            lbl = 'Closed'
        break

    if np.logical_and(np.all(rpred) == 0, np.all(lpred) == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 20:
        try:
            sound.play()
        except:
            pass
    if score >40 and not sms_sent:    
        # Drowsiness detected, send an emergency SMS
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        client = Client(account_sid, auth_token)
        for contact in emergency_contacts:
            message = client.messages.create(
                body="Emergency! Driver is drowsy. Location: {location}",
                from_=twilio_phone_number,
                to=contact
            )
        sms_sent = True

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
