import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

img_path = "/home/jasmeet/PycharmProjects/Character_Recognition/Tesseract/Text_recognition_with_EAST/test_image4.png"
model_path = "/home/jasmeet/PycharmProjects/Character_Recognition/Tesseract/frozen_east_text_detection.pb"
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
min_confi = 0.5
nw = 640
nh = 640

img = cv2.imread(img_path)
orig = img.copy()
net = cv2.dnn.readNet(model_path)
(h, w) = img.shape[:2]
rw = w/float(nw)
rh = h/float(nh)

img = cv2.resize(img, (nw, nh))
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rows, columns) = scores.shape[2:4]
rects = []
confidences = []

for r in range(0, rows):
    scoresData = scores[0, 0, r]
    xData0 = geometry[0, 0, r]
    xData1 = geometry[0, 1, r]
    xData2 = geometry[0, 2, r]
    xData3 = geometry[0, 3, r]
    anglesData = geometry[0, 4, r]

    for c in range(0, columns):
        print(scoresData[c])
        if scoresData[c] < min_confi:
            continue
        (offsetx, offsety) = (c * 4.0, r * 4.0)
        h = xData0[c] + xData2[c]
        w = xData1[c] + xData3[c]

        angle = anglesData[c]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetx + (cos * xData1[c]) + (sin * xData2[c]))
        endY = int(offsetx - (sin * xData1[c]) + (cos * xData2[c]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[c])

boxes = non_max_suppression(np.array(rects), probs=confidences)

for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rw)
    startY = int(startY * rh)
    endX = int(endX * rw)
    endY = int(endY * rh)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)


orig = cv2.resize(orig, (orig.shape[1], orig.shape[0]))
cv2.imshow("img", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()