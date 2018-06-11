import zipfile
import shutil
import os
import cv2
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from upscale import upsample
from tqdm import tqdm

#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0

    #---------------------------------------------------------------------------
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
def reshape(x, num_classes, upscale_factor, name):
    """
    Reshape the tensor so that it matches the number of classes and output size
    :param x:              input tensor
    :param num_classes:    number of classes
    :param upscale_factor: scaling factor
    :param name:           name of the resulting tensor
    :return:               reshaped tensor
    """
    with tf.variable_scope(name):
        
        w_shape = [1, 1, int(x.get_shape()[3]), num_classes]
        w = tf.Variable(tf.truncated_normal(w_shape, 0, 0.1),
                        name=name+'_weights')
        b = tf.Variable(tf.zeros(num_classes), name=name+'_bias')
        resized = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID',
                               name=name+'_resized')
        resized = tf.nn.bias_add(resized, b, name=name+'_add_bias')

    upsampled = upsample(resized, num_classes, upscale_factor,
                            name+'_upsampled')
    
    return upsampled

#-------------------------------------------------------------------------------
class FCNVGG:
    #---------------------------------------------------------------------------
    def __init__(self, session, rd_feature=None):
        self.session     = session
        if not rd_feature == None:
            self.rd_feature = rd_feature
        

    #---------------------------------------------------------------------------
    def build_from_vgg(self, vgg_dir, num_classes, progress_hook):
        """
        Build the model for training based on a pre-define vgg16 model.
        :param vgg_dir:       directory where the vgg model should be stored
        :param num_classes:   number of classes
        :param progress_hook: a hook to show download progress of vgg16;
                              the value may be a callable for urlretrieve
                              or string "tqdm"
        """
        self.num_classes = num_classes
        self.__download_vgg(vgg_dir, progress_hook)
        self.__load_vgg(vgg_dir)
        self.__make_result_tensors()

    #---------------------------------------------------------------------------
    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        """
        Build the model for inference from a metagraph shapshot and weights
        checkpoint.
        """
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
        self.rd_feature_train  = sess.graph.get_tensor_by_name('rd_feature_train:0')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')

        self.vgg_layer1  = sess.graph.get_tensor_by_name('pool1:0')
        self.vgg_layer2  = sess.graph.get_tensor_by_name('pool2:0')
        self.vgg_layer3  = sess.graph.get_tensor_by_name('pool3:0')

        self.vgg_layer4  = sess.graph.get_tensor_by_name('layer4_out:0')
        self.logits      = sess.graph.get_tensor_by_name('sum/Add_1:0')
        self.softmax     = sess.graph.get_tensor_by_name('result/Softmax:0')
        self.classes     = sess.graph.get_tensor_by_name('result/ArgMax:0')

    #---------------------------------------------------------------------------
    def __download_vgg(self, vgg_dir, progress_hook):
        #-----------------------------------------------------------------------
        # Check if the model needs to be downloaded
        #-----------------------------------------------------------------------
        vgg_archive = 'vgg.zip'
        vgg_files   = [
            vgg_dir + '/variables/variables.data-00000-of-00001',
            vgg_dir + '/variables/variables.index',
            vgg_dir + '/saved_model.pb']

        missing_vgg_files = [vgg_file for vgg_file in vgg_files \
                             if not os.path.exists(vgg_file)]

        if missing_vgg_files:
            if os.path.exists(vgg_dir):
                shutil.rmtree(vgg_dir)
            os.makedirs(vgg_dir)

            #-------------------------------------------------------------------
            # Download vgg
            #-------------------------------------------------------------------
            url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip'
            if not os.path.exists(vgg_archive):
                if callable(progress_hook):
                    urlretrieve(url, vgg_archive, progress_hook)
                else:
                    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                        urlretrieve(url, vgg_archive, pbar.hook)

            #-------------------------------------------------------------------
            # Extract vgg
            #-------------------------------------------------------------------
            zip_archive = zipfile.ZipFile(vgg_archive, 'r')
            zip_archive.extractall(vgg_dir)
            zip_archive.close()

    #---------------------------------------------------------------------------
    def __load_vgg(self, vgg_dir):
        sess = self.session
        graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir+'/vgg')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        # self.rd_input    = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')

        self.vgg_layer1  = sess.graph.get_tensor_by_name('pool1:0')
        self.vgg_layer2  = sess.graph.get_tensor_by_name('pool2:0')
        self.vgg_layer3  = sess.graph.get_tensor_by_name('layer3_out:0')
        
        # 输出所有operation的name
        # self.vgg_layer4  = sess.graph.get_tensor_by_name('layer4_out:0')
        # self.vgg_layer7  = sess.graph.get_tensor_by_name('layer7_out:0')
        # op = sess.graph.get_operations()
        # tens = [m.values() for m in op]
        # for each in tens:
        #     print(each)

    #---------------------------------------------------------------------------
    def __make_result_tensors(self):
        # TODO: 对VGG3 4 7 叠加RD特征，然后再上采样
        # self.rd_down2 = tf.nn.pool(self.rd_feature, window_shape=[2, 2], strides=[2, 2], pooling_type="MAX", padding="VALID")
        # self.rd_down4 = tf.nn.pool(self.rd_down2, window_shape=[2, 2], strides=[2, 2], pooling_type="MAX", padding="VALID")
        # self.rd_down8 = tf.nn.pool(self.rd_down4, window_shape=[2, 2], strides=[2, 2], pooling_type="MAX", padding="VALID")
        # self.rd_down16 = tf.nn.pool(self.rd_down8, window_shape=[2, 2], strides=[2, 2], pooling_type="MAX", padding="VALID")

        # self.new_vgg_layer3 = tf.concat([self.vgg_layer3, self.rd_down8], axis=3)
        # self.new_vgg_layer4 = tf.concat([self.vgg_layer4, self.rd_down16], axis=3)
        # new_vgg_layer7 = tf.concat([self.vgg_layer7, rd_down32], axis=3)
        
        vgg1_reshaped = reshape(self.vgg_layer1, self.num_classes,  2,
                                'layer1_resize')
        vgg2_reshaped = reshape(self.vgg_layer2, self.num_classes,  4,
                                'layer2_resize')
        vgg3_reshaped = reshape(self.vgg_layer3, self.num_classes,  8,
                                'layer3_resize')
        # vgg4_reshaped = reshape(self.vgg_layer4, self.num_classes, 16,
        #                         'layer4_resize')
        # vgg7_reshaped = reshape(self.vgg_layer7, self.num_classes, 32,
        #                         'layer7_resize')
            
        with tf.variable_scope('sum'):
            self.logits  = tf.add(tf.add(vgg1_reshaped, vgg2_reshaped), vgg3_reshaped)
            w_shape = [3, 3, self.num_classes, self.num_classes]
            w = tf.Variable(tf.truncated_normal(w_shape, 0, 0.1))
            b = tf.Variable(tf.zeros(self.num_classes))
            sum_conv1 = tf.nn.conv2d(self.logits, w, strides=[1, 1, 1, 1], padding='SAME')
            sum_conv1 = tf.nn.bias_add(sum_conv1, b)
            # sum_conv1 = tf.nn.dropout(sum_conv1, 0.75)
            # sum_conv1 = tf.nn.relu(sum_conv1)
            w2 = tf.Variable(tf.truncated_normal(w_shape, 0, 0.1))
            b2 = tf.Variable(tf.zeros(self.num_classes))
            sum_conv2 = tf.nn.conv2d(sum_conv1, w, strides=[1, 1, 1, 1], padding='SAME')
            sum_conv2 = tf.nn.bias_add(sum_conv2, b)
            # sum_conv2 = tf.nn.dropout(sum_conv2, 0.75)
            # sum_conv2 = tf.nn.relu(sum_conv2)
            self.logits   =  sum_conv2
        with tf.name_scope('result'):
            self.softmax  = tf.nn.softmax(self.logits)
            self.classes  = tf.argmax(self.softmax, axis=3)
    #---------------------------------------------------------------------------
    def get_optimizer(self, labels, learning_rate=0.0001):
        with tf.variable_scope('reshape'):
            labels_reshaped  = tf.reshape(labels, [-1, self.num_classes])
            logits_reshaped  = tf.reshape(self.logits, [-1, self.num_classes])
            softmax_reshaped = tf.reshape(self.softmax, [-1, self.num_classes])
            # TODO：FM的实现有BUG
            '''
            F-Measure Loss 
            '''
            softmax_result  = tf.nn.softmax(logits_reshaped)
            Y = tf.one_hot(tf.reshape(tf.argmax(softmax_result, axis=1), [-1]), depth = 2)
            P = tf.metrics.precision(labels_reshaped, Y)
            R = tf.metrics.recall(labels_reshaped, Y)
            fm = tf.reduce_mean(2 * tf.multiply(R[0], P[0]) / (R[0] + P[0]))

            '''
            mse loss
            '''
            
            losses_mse = tf.losses.mean_squared_error(labels_reshaped, softmax_reshaped)
            loss_mse = tf.reduce_mean(losses_mse)

            # loss = (loss_mse)
            losses_ce = tf.nn.softmax_cross_entropy_with_logits(labels=labels_reshaped, logits=logits_reshaped)
            loss_ce = tf.reduce_mean(losses_ce)
            '''
            PSNR
            '''
            psnr = 10 * self.tensor_log10(1.0 / loss_mse) 
            
        with tf.variable_scope('optimizer'):
            optimizer       = tf.train.AdamOptimizer(learning_rate)
            # optimizer       = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer       = optimizer.minimize(loss_ce)

        return optimizer, loss_mse, loss_ce, psnr, fm

    def tensor_log10(self, x):
        # log2转log10
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator