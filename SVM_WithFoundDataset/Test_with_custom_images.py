import cv2
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np

count = 0
clf = joblib.load("alphabet_classifier.pkl")


for i in range(122):
    img = cv2.imread("Testing/"+str(i+1)+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #retval, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, -1)
    prediction = clf.predict(img)
    print("For "+str(i+1)+".jpg:")
    if i+1 <= 88:
        if prediction == 0:
            count = count + 1
    if i+1 >= 89:
        if prediction == 2:
            count = count + 1
    print(prediction)


print("Accuracy: ")
print(count/122)