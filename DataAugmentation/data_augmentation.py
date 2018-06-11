import cv2
import os
import random

def resize_to_32(img, scale=1):
    if img.shape[0] % 32 < 16:
        new_height = img.shape[0] - img.shape[0] % 32
    else:
        new_height = img.shape[0] + 32 - img.shape[0] % 32

    if img.shape[1] % 32 < 16:
        new_width = img.shape[1] - img.shape[1] % 32
    else:
        new_width = img.shape[1] + 32 - img.shape[1] % 32

    new_size = (new_width*int(scale), new_height*int(scale))
    return cv2.resize(img, new_size)

def crop(img, roi):
    '''
    roi:        (x, y, width, height)
    '''
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    return img
    

def augment(img, width, height, gt_img):
    '''
    width:      输出图像的宽度
    height:     输出图像的高度
    '''
    # if img == None or gt_img == None:
    #     raise Exception('img and gt_img cannot be NoneType')
    for i in range(25): #取5个不同的原点
        random_x = random.random()
        random_y = random.random()
        range_x = img.shape[1] - width - 10 # 原点在横轴上的取值范围
        range_y = img.shape[0] - height - 10 # 原点在纵轴上的取值范围
        for scale in range(1, 4):
            img_scaled = resize_to_32(img, scale=scale)
            gt_img_scaled = resize_to_32(gt_img, scale=scale)
            origin_x = int(random_x*scale*range_x)
            origin_y = int(random_y*scale*range_y)
            img_augmented = crop(img_scaled , (origin_x, origin_y, width, height))
            cv2.imwrite('image' + '/' + str(i+1) + '_' + str(scale) + '_' + os.path.basename(path), img_augmented)
            gt_img_augmented = crop(gt_img_scaled, (origin_x, origin_y, width, height))
            cv2.imwrite('gt_image' + '/' + str(i+1) + '_' + str(scale) + '_' + os.path.splitext(os.path.basename(path))[0] + '_gt.png', gt_img_augmented)
        


path = input()
img = cv2.imread(path)
gt_path = input()
gt_img = cv2.imread(gt_path)
augment(img, width=224, height=224, gt_img=gt_img)
