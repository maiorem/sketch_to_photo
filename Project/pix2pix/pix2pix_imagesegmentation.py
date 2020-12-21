import numpy as np
import cv2 as cv
from scipy.ndimage import label
import os
import glob

def image_seg(img) :
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 노이즈 삭제
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)


    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    result_dist_transform = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(),255, cv.THRESH_BINARY)


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)


    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0


    markers = cv.watershed(img, markers)

    img[markers == -1] = [255, 0, 0]
    img[markers == 1] = [255, 255, 0]

    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


a = glob.glob("./data/photo/*/*")

photo_idx = []
for i in range(len(a)):

    c = a[0]+"/"+a[1]
    print(c)

    photo_idx.append(c)

print(photo_idx)
print(len(photo_idx)) #276

for idx in photo_idx :
    img = cv.imread(idx)
    image_seg(img)