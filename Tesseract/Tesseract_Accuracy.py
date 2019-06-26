import cv2
import pytesseract
import os
from PIL import Image


count = 0
accuracy = 0


for x in range(88):
    img = cv2.imread("A_testing/"+str(x+1)+".jpg", 0)
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(str(x+1)+".png", img)
    text = pytesseract.image_to_string(Image.open(str(x+1)+".png"), lang='eng', config='--psm 10')
    print(text)
    if text == 'A' or text == 'a':
        count = count + 1

    os.remove(str(x+1)+".png")


print(count)
accuracy = float(count/88)
print("Accuracy is:  ")
print(accuracy)