import random
import cv2
import os
import numpy as np
from glob import glob
from relative_darkness import relative_darkness

#-------------------------------------------------------------------------------

class DIBCOSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.image_size = (224, 224)
        self.num_classes = 2
        self.label_colors = {0: np.array([0, 0, 0]),
                             1: np.array([255, 255, 255])}

        self.num_training = None
        self.num_validation = None
        self.train_generator = None
        self.valid_generator = None

    #---------------------------------------------------------------------------
    def load_data(self, data_dir, total_fraction, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        images = data_dir + '/sample/image/*.png'
        labels = data_dir + '/sample/gt_image/*_gt.png'

        image_paths = glob(images)
        label_paths = {
            os.path.basename(path).replace('_gt', ''): path
            for path in glob(labels)}

        self.label_paths = label_paths

        num_images = len(image_paths)
        if num_images == 0:
            raise RuntimeError('No data files found in ' + data_dir)

        random.shuffle(image_paths)
        image_paths = image_paths[:int(total_fraction*num_images)]
        num_images = len(image_paths)
        random.shuffle(image_paths)
        valid_images = image_paths[:int(valid_fraction*num_images)]
        train_images = image_paths[int(valid_fraction*num_images):]

        self.num_training = len(train_images)
        self.num_validation = len(valid_images)
        self.num_classes = 2
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)

    #---------------------------------------------------------------------------
    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):
            background_color = np.array([255, 255, 255])

            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset+batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []

                p_count = 1
                for image_file in files:
                    label_file = self.label_paths[os.path.basename(image_file)]
                    image = cv2.resize(cv2.imread(image_file), self.image_size)
                    label = cv2.resize(cv2.imread(label_file), self.image_size)

                    label_bg = np.all(label == background_color, axis=2)
                    label_fg = np.any(label != background_color, axis=2)
                    label_all = np.dstack([label_fg, label_bg])
                    label_all = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        # names_rds.append(rd_features)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                        names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)

        return gen_batch

#-------------------------------------------------------------------------------


def get_source():
    return DIBCOSource()
