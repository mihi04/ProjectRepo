import cv2
import numpy as np

haar_data = cv2.CascadeClassifier('C:\\Users\\91998\\projects\\mask detection\\data.xml')


capture = cv2.VideoCapture(0)  #0 to detect the default camera
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)  
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
        print(cv2.imshow('result', img))
        # 27 is the ASCII value of the escape button
        if cv2.waitKey(2) == 27 or len(data)>=200:   #the funct waitKey waits for a key infinitely or for a delay
            break
capture.release()
cv2.destroyAllWindows()

np.save('with_mask.npy', data)