import cv2 as cv
import numpy as np

from data_set import pred


def distance_pixel(x, y):
    r_mean = (x[2] + y[2])/2
    return ((2 + (255 - r_mean)/255) * (x[0] - y[0])**2 + 4 * (x[1] - y[1])**2 + (2 + r_mean/255) * (x[2] - y[2])**2)**0.5


def distance_count(clr, img):
    clr = clr.reshape(5,3)
    distance_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distance_cl_list = []
            for k in range(5):
                distance_cl_list.append(distance_pixel(img[i][j], clr[k]))
            distance_list.append(np.min(distance_cl_list))
    return np.mean(distance_list)


col = np.array([[234.88073179380257, 250.0879701331056, 253.40816722060174], [87.73979957316551, 116.806765564079, 143.29058966136773], [74.8288885872779, 89.13647741853356, 97.40532997182783], [65.14808334932438, 125.1647668302534, 219.67886072113663], [44.78690186149352, 49.52885222846116, 51.81857709079451]]
               )

img = cv.imread('test/7.jpg')

print(pred(col), distance_count(col,img))