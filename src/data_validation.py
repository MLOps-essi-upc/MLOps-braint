import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os


directory = os.fsencode('data/raw')
def is_noisy(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.01
    return s_perc > s_thr

if __name__ == "__main__":

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if '.jpg' in file.decode():
                path=subdir.decode()+"\\"+file.decode()
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if is_noisy(img):
                    img = cv2.medianBlur(img, 3)
                    cv2.imwrite(path,img)

#write in the directory or DVC?
