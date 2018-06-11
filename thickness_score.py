import numpy as np
import numpy.ma as ma
import os
import cv2
import copy
import scipy.ndimage as nd
from skimage.filters import threshold_sauvola

# 估计笔画粗细值
def thickness_score(im, window_size=5, threshold=10, name=None):
    if im.ndim == 3:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('raw_' + name, im)
    
    edge = cv2.Canny(im,50,100)  
    cv2.imwrite('edge_' + name, edge)
    e_pixel_count = 0
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if edge[i][j] == 255:
                e_pixel_count += 1
    

    # 计算感兴趣的区域（文字区域）
    # threshold_val, _ = cv2.threshold(im, -1, 255, cv2.THRESH_OTSU)
    threshold_val = threshold_sauvola(im, window_size=15, k=0.15)
    roi = np.zeros_like(im)
    roi = np.array(im < threshold_val + 10).astype(np.int32) # roi范围扩大一些？
    roi_pixel_count = 0
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] == 1:
                roi_pixel_count += 1

    cv2.imwrite('roi_' + name, roi*255)

    stroke_width = 0

    if e_pixel_count == 0:
        stroke_width = -1
    else:
        stroke_width =  float(roi_pixel_count / e_pixel_count)
    print('stroke width:', stroke_width)
    return stroke_width
    
