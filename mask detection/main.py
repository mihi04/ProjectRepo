# import matplotlib.pyplot as plt

# plt.imshow(data[0])

import numpy as np
import cv2

with_mask = np.load('C:\\Users\\91998\\projects\\mask detection\\with_mask.npy')
without_mask = np.load('C:\\Users\\91998\\projects\\mask detection\\without_mask.npy')

with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)

print(with_mask.shape)
print(without_mask.shape)

X = np.r_[with_mask, without_mask]

print(X.shape)

labels = np.zeros(X.shape[0])
labels[200:] = 1.0

names = {0 : 'Mask', 1: 'No Mask' }

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)

print(x_train.shape)

#dimentionality reduction of large data set to make the model faster
# PCA - principal component analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

# print(x_train[0])

print(x_train.shape)

svm = SVC()
svm.fit(x_train, y_train)

x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))


haar_data = cv2.CascadeClassifier('C:\\Users\\91998\\projects\\mask detection\\data.xml')

capture = cv2.VideoCapture(0)  #0 to detect the default camera
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]  
            face = cv2.resize(face, (50,50))
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250, 250), 2)
            print(n)
        
        print(cv2.imshow('result', img))
        # 27 is the ASCII value of the escape button
        if cv2.waitKey(2) == 27:   #the funct waitKey waits for a key infinitely or for a delay
            break
capture.release()
cv2.destroyAllWindows()



