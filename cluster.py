import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as sco
import cv2 as cv2
import os
import glob
import time
import itertools
from sklearn.cluster import KMeans
from PIL import Image

from data_set import pred


image_path = './test'
K = 5
alpha = 0.05
paths = glob.glob(os.path.join(image_path, '*.jpg'))


def show_color_knn(ImgIdx, title):
    x = 1000
    y = 1000
    c = Image.new("RGB", (x, y))
    for i in range(0, x):
        for j in range(0, y):
            if i < 200:
                c.putpixel([i, j], (int(ImgIdx[0][0]), int(ImgIdx[0][1]), int(ImgIdx[0][2])))
            elif i < 400:
                c.putpixel([i, j], (int(ImgIdx[1][0]), int(ImgIdx[1][1]), int(ImgIdx[1][2])))
            elif i < 600:
                c.putpixel([i, j], (int(ImgIdx[2][0]), int(ImgIdx[2][1]), int(ImgIdx[2][2])))
            elif i < 800:
                c.putpixel([i, j], (int(ImgIdx[3][0]), int(ImgIdx[3][1]), int(ImgIdx[3][2])))
            elif i < 1000:
                c.putpixel([i, j], (int(ImgIdx[4][0]), int(ImgIdx[4][1]), int(ImgIdx[4][2])))
    c.save('./result05/' + title + '_knn.jpg')


def show_color_seg(ImgIdx, title):
    x = 1000
    y = 1000
    c = Image.new("RGB", (x, y))
    for i in range(0, x):
        for j in range(0, y):
            if i < 200:
                c.putpixel([i, j], (int(ImgIdx[0][2]), int(ImgIdx[0][1]), int(ImgIdx[0][0])))
            elif i < 400:
                c.putpixel([i, j], (int(ImgIdx[1][2]), int(ImgIdx[1][1]), int(ImgIdx[1][0])))
            elif i < 600:
                c.putpixel([i, j], (int(ImgIdx[2][2]), int(ImgIdx[2][1]), int(ImgIdx[2][0])))
            elif i < 800:
                c.putpixel([i, j], (int(ImgIdx[3][2]), int(ImgIdx[3][1]), int(ImgIdx[3][0])))
            elif i < 1000:
                c.putpixel([i, j], (int(ImgIdx[4][2]), int(ImgIdx[4][1]), int(ImgIdx[4][0])))
    c.save('./result05/' + title + '_seg.jpg')


def distance_pixel(x, y):
    r_mean = (x[2] + y[2])/2
    return ((2 + (255 - r_mean)/255) * (x[0] - y[0])**2 + 4 * (x[1] - y[1])**2 + (2 + r_mean/255) * (x[2] - y[2])**2)**0.5


def knn(path):
    class_list = []
    img = mpimg.imread(path)
    imgs = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=K,).fit(imgs)
    for i in img:
        class_list.append(kmeans.predict(i))
    distance_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distance_list.append(distance_pixel(img[i][j], kmeans.cluster_centers_[class_list[i][j]]))
    return kmeans.cluster_centers_.reshape(15), np.mean(distance_list)


def seg(path):
    class_list = []
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=100, min_size=50)
    src = cv2.imread(path)
    segment = segmentator.processImage(src)
    mask = segment.reshape(list(segment.shape) + [1]).repeat(3, axis=2)
    masked = np.ma.masked_array(src, fill_value=0)
    knn_o = []
    for i in range(np.max(segment)):
        masked.mask = mask != i
        y, x = np.where(segment == i)
        li = np.zeros((y.shape[0], 3))
        for j in range(y.shape[0] - 1):
            li[j] = src[y[j], x[j]]
        knn_o.append(np.average(li, axis=0))
    knn_o = np.asarray(knn_o)
    kmeans = KMeans(n_clusters=K).fit(knn_o)
    for i in src:
        class_list.append(kmeans.predict(i))
    distance_list = []
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            distance_list.append(distance_pixel(src[i][j], kmeans.cluster_centers_[class_list[i][j]]))
    return kmeans.cluster_centers_.reshape(15), np.mean(distance_list)


def distance_count(clr, img):
    clr = clr.reshape(5, 3)
    distance_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distance_cl_list = []
            for k in range(5):
                distance_cl_list.append(distance_pixel(img[i][j], clr[k]))
            distance_list.append(np.min(distance_cl_list))
    return np.mean(distance_list)


def fuction2op(clr):
    return distance_count(clr, img)*alpha - pred(clr)


def callbackf(x):
    print(fuction2op(x))
    print(np.array2string(x, separator=', '))
    print(time.ctime())


def max_point(clr):
    max_p = 0
    clr_per = clr
    for i in list(itertools.permutations([0, 1, 2, 3, 4])):
        hi_clr = np.array([clr[i[0]], clr[i[1]], clr[i[2]], clr[i[3]], clr[i[4]]])
        if max_p < pred(hi_clr):
            max_p = pred(hi_clr)
            clr_per = hi_clr
    return clr_per

for path in paths:
    title = path.split('\\')
    title = title[1].split('.')[0]
    print('-----------image '+title + ' is start-----------')
    #knn_res = knn(path)
    seg_res = seg(path)
    #show_color_knn(knn_res[0], title)
    #print(pred(knn_res[0]), knn_res[1])
    show_color_seg(seg_res[0].reshape(5, 3), title)
    #print(pred(seg_res[0]), seg_res[1])
    img = cv2.imread(path)
    print('start point and distance:', fuction2op(seg_res[0]), seg_res[1])
    print('start estimate:', pred(seg_res[0]))
    print('seg result color:')
    print(np.array2string(seg_res[0], separator=', '))
    print('start time:', time.ctime())
    print('----------  optimize start  -----------')
    inp = max_point(seg_res[0].reshape(5, 3))
    res = sco.fmin(fuction2op, inp, maxiter=50)
    print('optimize finish time:', time.ctime())
    print('---------- optimize finish ------------')
    res = np.array(res)
    res = res.reshape(5, 3)
    print('optimized point and distance:', fuction2op(res), distance_count(res, img))
    print('optimized estimate:', pred(res))
    res = res.tolist()
    show_color_seg(res, title+'_opt')
    print('optimized result:')
    print(res)
    print('------------------image '+title + ' is over------------------')


'''title = paths[0].split('\\')
title = title[1].split('.')[0]
#knn_res = knn(paths[0])
seg_res = seg(paths[0])
img = cv2.imread(paths[0])
print('start point:', fuction2op(seg_res[0]))
print('knn result color:', np.array2string(seg_res[0], separator=', '))
print('start time:', time.ctime())
print('----------optimize start-----------')
res = sco.fmin(fuction2op, seg_res[0],  maxiter = 30)
print('----------optimize finish------------')
print(res)'''
#show_color_knn(knn_res[0], title)
#print(pred(knn_res[0]), knn_res[1])
#show_color_seg(seg_res[0], title)
#print(pred(seg_res[0]), seg_res[1])
#opt = sco.minimize_scalar(fun=fuction2op, )
#print('------------------'+title + 'is over------------------')

