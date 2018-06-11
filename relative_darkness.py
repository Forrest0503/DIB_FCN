import numpy as np
import os
import cv2
import scipy.ndimage as nd

def relative_darkness(im, window_size=5, threshold=10):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# find number of pixels at least $threshold less than the center value
    def below_thresh(vals):
        center_val = vals[int(vals.shape[0]/2)]
        lower_thresh = center_val - threshold
        return (vals < lower_thresh).sum()

	# find number of pixels at least $threshold greater than the center value
    def above_thresh(vals):
        center_val = vals[int(vals.shape[0]/2)]
        above_thresh = center_val + threshold
        return (vals > above_thresh).sum()
		
	# apply the above function convolutionally
    lower = nd.generic_filter(im, below_thresh, size=window_size, mode='reflect')
    upper = nd.generic_filter(im, above_thresh, size=window_size, mode='reflect')

	# number of values within $threshold of the center value is the remainder
	# constraint: lower + middle + upper = window_size ** 2
    middle = np.empty_like(lower)
    middle.fill(window_size*window_size)
    middle = middle - (lower + upper)

	# scale to range [0-255]
    lower = lower * (255 / (window_size * window_size))
    middle = middle * (255 / (window_size * window_size))
    upper = upper * (255 / (window_size * window_size))

    return np.concatenate( [lower[:,:,np.newaxis], middle[:,:,np.newaxis], upper[:,:,np.newaxis]], axis=2)

def get_RGBA(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    rd_img = relative_darkness(img)
    data = np.concatenate([img[:,:,np.newaxis], rd_img], axis=2)
    cv2.imwrite(os.path.basename(path), rd_img)

# for i in range(1, 21):
#     try:
#         get_RGBA('/Users/Jason/Developer/ML/image-segmentation-fcn-master/data/training_RD/image/' + str(i) + '.png')
#     except:
#         pass