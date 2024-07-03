import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\91998\\projects\\mask detection\\face.jpg')

# print(img.shape)

# print(img[0])

# print(img)


# p = plt.imshow(img)  #it removes the colour of the img and just shows its axis
# print(p)


# while True:
#     print(cv2.imshow('result', img))
#     # 27 is the ASCII value of the escape button
#     if cv2.waitKey(2) == 27:   #the funct waitKey waits for a key infinitely or for a delay
#         break
# cv2.destroyAllWindows()

haar_data = cv2.CascadeClassifier('C:\\Users\\91998\\projects\\mask detection\\data.xml')

a = haar_data.detectMultiScale(img)
print(a)   #prints array([[x, y, w, h]]) of the haar features of the image

# cv2.rectangle(img, (x,y), (w,h), (b,g,r), border_thickness)

## detecting Faces

# while True:
#     faces = haar_data.detectMultiScale(img)
#     for x, y, w, h in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)  
        
#     print(cv2.imshow('result', img))
#     # 27 is the ASCII value of the escape button
#     if cv2.waitKey(2) == 27:   #the funct waitKey waits for a key infinitely or for a delay
#         break
# cv2.destroyAllWindows()


##colecting data for mask detection
##code to detect the face using front camera

# capture = cv2.VideoCapture(0)  #0 to detect the default camera
# while True:
#     flag, img = capture.read()
#     if flag:
#         faces = haar_data.detectMultiScale(img)
#         for x, y, w, h in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)  
        
#         print(cv2.imshow('result', img))
#         # 27 is the ASCII value of the escape button
#         if cv2.waitKey(2) == 27:   #the funct waitKey waits for a key infinitely or for a delay
#             break
# capture.release()
# cv2.destroyAllWindows()

###learning to use numpy

import numpy as np

x = np.array([3, 4, 5, 6])
# print(x)

# print(x[0:2] ) #slicing

y = np.array([[3, 4, 5, 6], [4, 5, 7, 8], [4, 5, 6, 8]])
# print(y)

#slicing in two dimensional array

# print(y[0])

# print(y[0,1:4])

#y[row, col]

# print(y[0:3, 1:3])

# z = y[:,1:2]  #All rows but some column
# print(z)

#***************************

# capture = cv2.VideoCapture(0)  #0 to detect the default camera
# data = []
# while True:
#     flag, img = capture.read()
#     if flag:
#         faces = haar_data.detectMultiScale(img)
#         for x, y, w, h in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)  
#             face = img[y:y+h, x:x+w, :]
#             face = cv2.resize(face, (50,50))
#             print(len(data))
#             if len(data)<400:
#                 data.append(face)
#         print(cv2.imshow('result', img))
#         # 27 is the ASCII value of the escape button
#         if cv2.waitKey(2) == 27 or len(data)>=200:   #the funct waitKey waits for a key infinitely or for a delay
#             break
# capture.release()
# cv2.destroyAllWindows()

# np.save('without_mask.npy', data)







