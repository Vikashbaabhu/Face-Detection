#Face Detection 

import cv2
fc = cv2.CascadeClassifier('front_face.xml')
img = cv2.imread('face.jpg')                    # Reads the image 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # Converts the image to grayscale

face = fc.detectMultiScale(gray,1.1,9)          # Tuning parameters

for x,y,w,h in face:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

cv2.imshow('FACE DETECTION',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
