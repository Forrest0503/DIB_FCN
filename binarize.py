import argparse
import math
import sys
import cv2
import os
import tensorflow as tf
import numpy as np
from fcnvgg import FCNVGG
from utils import *
from glob import glob
from tqdm import tqdm
from relative_darkness import relative_darkness
from thickness_score import thickness_score
from tensorflow.python import pywrap_tensorflow
import matplotlib as plt

#-------------------------------------------------------------------------------
PATCH_SIZE = 224

def get_subwindows(img):
    # 边缘用平均值填充，而不是用255填充
    subwindows = []
    positions = []

    height = img.shape[0]
    width = img.shape[1]
    if not img.shape[0] % PATCH_SIZE == 0:
        height = img.shape[0] - img.shape[0] % PATCH_SIZE + PATCH_SIZE
    if not img.shape[1] % PATCH_SIZE == 0:
        width = img.shape[1] - img.shape[1] % PATCH_SIZE + PATCH_SIZE
    
    expanded_img = np.zeros((height, width, 3))
    for row in range(height):
        for col in range(width):
            if row >= img.shape[0] or col >= img.shape[1]:
                if col == img.shape[1] and row < img.shape[0]:
                    average = [0, 0, 0]
                    for c in range(3):
                        for i in range(col-5, col):
                            average[c] += img[row][i][c]
                        average[c] /= 5.0
                        expanded_img[row][col][c] = average[c] # 使用均值填充
                elif col > img.shape[1] and row < img.shape[0]:
                    for c in range(3):
                        expanded_img[row][col][c] = expanded_img[row][img.shape[1]][c]

                if row == img.shape[0]:
                    average = [0, 0, 0]
                    for c in range(3):
                        for i in range(row-5, row):
                            average[c] += expanded_img[i][col][c]
                        average[c] /= 5.0
                        expanded_img[row][col][c] = average[c] # 使用均值填充
                elif row > img.shape[0]:
                    for c in range(3):
                        expanded_img[row][col][c] = expanded_img[img.shape[0]][col][c]
            else:
                for c in range(3):
                    expanded_img[row][col][c] = img[row][col][c]
    # cv2.imwrite('expanded.png', expanded_img)
    for y in range(int(height/PATCH_SIZE)):
        for x in range(int(width/PATCH_SIZE)):
            pos = (x, y)
            sub_img = expanded_img[y*PATCH_SIZE:(y+1)*PATCH_SIZE, x*PATCH_SIZE:(x+1)*PATCH_SIZE]
            subwindows.append(sub_img)
            positions.append(pos)
    
    return len(subwindows), zip(subwindows, positions)
    

def sample_generator(samples, image_size, batch_size, auto_scale=False, split=False):
    # for offset in range(0, len(samples), batch_size):
        # files = samples[offset:offset+batch_size]
    image_file = samples[0] # 一次只处理一张图片
    image = cv2.imread(image_file) 
    _, subwindows = get_subwindows(image)
    batch_count = 0
    for each in subwindows:
        window = each[0] 
        pos = each[1]
        patchs = [] # 每个batch只有一张图片
        patch_names = []
        batch_count += 1
        if auto_scale:
            score = thickness_score(window, name=str(batch_count)+'_'+os.path.basename(image_file))
            if score == -1:
                pass
            # 划分阈值
            if score < 1:
                window = cv2.resize(window, (896, 896))
            elif score >= 1 and score < 1.5:
                window = cv2.resize(window, (448, 448))
            elif score >= 1.5 and score < 2:
                window = cv2.resize(window, (256, 256))
            print('shape:', window.shape)
        else:
            pass

        patchs.append(window.astype(np.float32))            
        patch_names.append(str(pos[0])+'-'+str(pos[1])+'@'+os.path.basename(image_file))
        yield np.array(patchs), patch_names
        


#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='aws1',
                    help='project name')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--samples-dir', default='aws1',
                    help='directory containing samples to analyse')
parser.add_argument('--output-dir', default='test-output',
                    help='directory for the resulting images')
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')
parser.add_argument('--data-source', default='dibco',
                    help='data source')
parser.add_argument('--autoscale', default=False,
                    help='autoscale')
args = parser.parse_args()

#-------------------------------------------------------------------------------
# Check if we can get the checkpoint
#-------------------------------------------------------------------------------
state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'

if not os.path.exists(metagraph_file):
    print('[!] Cannot find metagraph ' + metagraph_file)
    sys.exit(1)

#-------------------------------------------------------------------------------
# Load the data source
#-------------------------------------------------------------------------------
try:
    source = load_data_source(args.data_source)
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Create a list of files to analyse and make sure that the output directory
# exists
#-------------------------------------------------------------------------------
samples = glob(args.samples_dir + '/*.png')
if len(samples) == 0:
    print('[!] No input samples found in', args.samples_dir)
    sys.exit(1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:      ', args.name)
print('[i] Network checkpoint:', checkpoint_file)
print('[i] Metagraph file:    ', metagraph_file)
print('[i] Number of samples: ', len(samples))
print('[i] Output directory:  ', args.output_dir)
print('[i] Image size:        ', source.image_size)
print('[i] # classes:         ', source.num_classes)
print('[i] Batch size:        ', args.batch_size)

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    print('[i] Creating the model...')

    rd_feature = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='rd_feature_infer')

    net = FCNVGG(sess, rd_feature)
    net.build_from_metagraph(metagraph_file, checkpoint_file)

    #---------------------------------------------------------------------------
    # Process the images
    #---------------------------------------------------------------------------
    generator = sample_generator(samples, source.image_size, args.batch_size, auto_scale=args.autoscale, split=True)
    # n_sample_batches = int(math.ceil(len(samples)/args.batch_size))
    n_sample_batches, _ = get_subwindows(cv2.imread(samples[0]))
    description = '[i] Processing samples'

    whole_img = np.zeros((cv2.imread(samples[0]).shape[0]*2, cv2.imread(samples[0]).shape[1]*2, 3)) 

    for x, names in tqdm(generator, total=n_sample_batches,
                            desc=description, unit='batches'):
        vgg_layer1  = sess.graph.get_tensor_by_name('pool3:0')
        feed = {net.image_input:  x,
                net.keep_prob:    1}
        layer1, img_labels = sess.run([vgg_layer1, net.classes], feed_dict=feed)
        imgs = draw_labels_batch(x, img_labels, label_colors, False)
        #---------------------------------------------------------------------------
        # 输出特征映射
        #---------------------------------------------------------------------------
        # chs = 256
        # range_stop = chs // 3
        # size_splits = [3 for i in range(0, range_stop)]
        # if len(size_splits) * 3 < chs:
        #     size_splits.append(chs % 3)
        # layer1_split = tf.split(layer1, num_or_size_splits=size_splits, axis=3)  # conv1.shape = [128,24,24,64]

        # layer1_concats_1 = []
        # concat_step = len(layer1_split) // 2
        # for i in range(0, concat_step, 2):
        #     concat = tf.concat([layer1_split[i], layer1_split[i + 1]], axis=1)
        #     layer1_concats_1.append(concat)

        # layer1_concats_2 = []
        # concat_step = len(layer1_concats_1) // 2
        # for i in range(0, concat_step, 2):
        #     concat = tf.concat([layer1_concats_1[i], layer1_concats_1[i + 1]], axis=2)
        #     layer1_concats_2.append(concat)

        # layer1_concats = tf.concat(layer1_concats_2, axis=0)

        # print(layer1_concats.shape)
        # layer1_np = layer1_concats.eval()
        # # print(layer1_np[0])
        # cv2.imwrite('featuremap' + names[0] + '.png', layer1_np[0])
        # print("visualize finish.")
        
        pos = names[0].split('@')[0]
        pos_x = int(pos.split('-')[0])
        pos_y = int(pos.split('-')[1])
            
        cv2.imwrite(args.output_dir + '/' + names[0], cv2.resize(imgs[0, :, :, :], (PATCH_SIZE, PATCH_SIZE)))
        whole_img[pos_y*PATCH_SIZE:(pos_y+1)*PATCH_SIZE, pos_x*PATCH_SIZE:(pos_x+1)*PATCH_SIZE, :] = \
                            cv2.resize(imgs[0, :, :, :], (PATCH_SIZE, PATCH_SIZE))

    whole_img = whole_img[0:cv2.imread(samples[0]).shape[0], 0:cv2.imread(samples[0]).shape[1]]
    cv2.imwrite(args.output_dir + '/' + 'whole_' + names[0], whole_img)

print('[i] All done.')
