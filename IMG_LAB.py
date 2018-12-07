import cv2
import numpy as np
vidcap = cv2.VideoCapture('1080.mp4')
success,image = vidcap.read()
count = 0
colorRBG = []
gray = []
colorLAB = []
while success:
    cy = image.shape[0] / 2
    cx = image.shape[1] / 2
    resize = min(cx, cy)
    crop_img = image[int(cy - resize):int(cy + resize), int(cx - resize):int(cx + resize)]
    resize_img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2LAB).astype("float32")
    colorLAB.append(lab_img)
    # cv2.imwrite("bwframe%d.jpg" % count, gray_img)
    colorRBG.append(resize_img)
    gray.append(gray_img)
    # cv2.imwrite("frame%d.jpg" % count, resize_img)     # save frame as JPEG file
    success,image = vidcap.read()

    # print('Read a new frame: ', success)
    count += 1
f = open("LAB.np","wb")
np.save(f,colorLAB)
# print(np.array(colorLAB).shape)