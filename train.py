#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import os

import tensorflow as tf
import numpy as np
import scipy.ndimage as nd

from fcnvgg import FCNVGG
from utils import *
from tqdm import tqdm
from relative_darkness import relative_darkness
import pandas as pd


#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the FCN')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--data-source', default='dibco',
                    help='data source')
parser.add_argument('--data-dir', default='data',
                    help='data directory')
parser.add_argument('--vgg-dir', default='vgg_graph',
                    help='directory for the VGG-16 model')
parser.add_argument('--epochs', type=int, default=70,
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=100,
                    help='checkpoint interval')
args = parser.parse_args()

print('[i] Project name:         ', args.name)
print('[i] Data source:          ', args.data_source)
print('[i] Data directory:       ', args.data_dir)
print('[i] VGG directory:        ', args.vgg_dir)
print('[i] # epochs:             ', args.epochs)
print('[i] Batch size:           ', args.batch_size)
print('[i] Tensorboard directory:', args.tensorboard_dir)
print('[i] Checkpoint interval:  ', args.checkpoint_interval)

try:
    print('[i] Creating directory {}...'.format(args.name))
    os.makedirs(args.name)
except (IOError) as e:
    print('[!]', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Configure the data source
#-------------------------------------------------------------------------------
print('[i] Configuring data source...')
try:
    source = load_data_source(args.data_source)
    source.load_data(args.data_dir, 1.0, 0.25)  # 第二个参数为验证集所占比例
    print('[i] # training samples:   ', source.num_training)
    print('[i] # validation samples: ', source.num_validation)
    print('[i] # classes:            ', source.num_classes)
    print('[i] Image size:           ', source.image_size)
    train_generator = source.train_generator
    valid_generator = source.valid_generator
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)


loss_list = []
val_loss_list = []
psnr_list = []
#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    print('[i] Creating the model...')
    rd_feature = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='rd_feature_train')
    net = FCNVGG(sess, rd_feature)
    net.build_from_vgg(args.vgg_dir, source.num_classes, progress_hook='tqdm')

    labels = tf.placeholder(tf.int32,
                            shape=[None, None, None, source.num_classes], name='labels_train')

    raw_image = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='raw_img_train')

    optimizer, loss_mse, loss_ce, psnr, fm = net.get_optimizer(labels)

    summary_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=10)

    label_mapper = tf.argmax(labels, axis=3)
    n_train_batches = int(math.ceil(source.num_training/args.batch_size))

    initialize_uninitialized_variables(sess)
    print('[i] Training...')

    #---------------------------------------------------------------------------
    # Summaries
    #---------------------------------------------------------------------------
    validation_loss = tf.placeholder(tf.float32, name='val_loss_train')
    validation_loss_summary_op = tf.summary.scalar('validation_loss',
                                                   validation_loss)

    training_loss = tf.placeholder(tf.float32, name='train_loss_train')
    training_loss_summary_op = tf.summary.scalar('training_loss',
                                                 training_loss)

    validation_img = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='val_img_train')
    validation_img_gt = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='val_img_gt_train')
    validation_img_summary_op = tf.summary.image('validation_img',
                                                 validation_img)
    validation_img_gt_summary_op = tf.summary.image('validation_img_gt',
                                                    validation_img_gt)
    validation_img_summary_ops = [validation_img_summary_op,
                                  validation_img_gt_summary_op]

    for e in range(args.epochs):
        #-----------------------------------------------------------------------
        # Train
        #-----------------------------------------------------------------------
        generator = train_generator(args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        training_loss_total_mse = 0
        training_loss_total_ce = 0
        fm_total = 0
        for x, y in tqdm(generator, total=n_train_batches,
                            desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    labels:           y,
                    net.keep_prob:    0.5}
            loss_batch_mse, loss_batch_ce, _ = sess.run([loss_mse, loss_ce, optimizer], feed_dict=feed)
            training_loss_total_mse += loss_batch_mse * \
                x.shape[0]
            training_loss_total_ce += loss_batch_ce * \
                x.shape[0]
        training_loss_total_mse /= source.num_training
        training_loss_total_ce /= source.num_training
        print("training_loss_total_mse: " + str(training_loss_total_mse))
        print("training_loss_total_ce: " + str(training_loss_total_ce))
        loss_list.append(training_loss_total_ce)

        #-----------------------------------------------------------------------
        # Validate
        #-----------------------------------------------------------------------
        generator = valid_generator(args.batch_size)
        validation_loss_total_mse = 0
        validation_loss_total_ce = 0
        psnr_total = 0
        imgs = None
        img_labels = None
        img_labels_gt = None
        for x, y in generator:
            feed = {net.image_input:  x,
                    labels:           y,
                    net.keep_prob:    1}
            loss_batch_mse, loss_batch_ce, img_classes, y_mapped, psnr_batch = sess.run([loss_mse, 
                                                          loss_ce,
                                                          net.classes,
                                                          label_mapper, 
                                                          psnr],
                                                         feed_dict=feed)
            validation_loss_total_mse += loss_batch_mse * x.shape[0]
            validation_loss_total_ce += loss_batch_ce * x.shape[0]
            psnr_total += psnr_batch * x.shape[0]
            if imgs is None:
                imgs = x[:3, :, :, :]
                img_labels = img_classes[:3, :, :]
                img_labels_gt = y_mapped[:3, :, :]

        validation_loss_total_mse /= source.num_validation
        validation_loss_total_ce /= source.num_validation
        psnr_total /= source.num_validation
        print("validaton_loss_total_mse: " + str(validation_loss_total_mse))
        print("validaton_loss_total_ce: " + str(validation_loss_total_ce))
        val_loss_list.append(validation_loss_total_ce)
        print("psnr_total: " + str(psnr_total))
        psnr_list.append(psnr_total)

        # 输出csv
        dataframe = pd.DataFrame({'loss':loss_list, 'val_loss':val_loss_list, 'psnr':psnr_list})
        dataframe.to_csv("losses.csv",index=False,sep=',')

        #-----------------------------------------------------------------------
        # Write loss summary
        #-----------------------------------------------------------------------
        feed = {validation_loss: validation_loss_total_ce,
                training_loss:   training_loss_total_ce}
        loss_summary = sess.run([validation_loss_summary_op,
                                 training_loss_summary_op],
                                feed_dict=feed)

        summary_writer.add_summary(loss_summary[0], e)
        summary_writer.add_summary(loss_summary[1], e)

        #-----------------------------------------------------------------------
        # Write image summary every 5 epochs
        #-----------------------------------------------------------------------
        if e % 5 == 0:
            imgs_inferred = draw_labels_batch(imgs, img_labels, label_colors)
            imgs_gt = draw_labels_batch(imgs, img_labels_gt, label_colors)

            feed = {validation_img:    imgs_inferred,
                    validation_img_gt: imgs_gt}
            validation_img_summaries = sess.run(validation_img_summary_ops,
                                                feed_dict=feed)
            summary_writer.add_summary(validation_img_summaries[0], e)
            summary_writer.add_summary(validation_img_summaries[1], e)

        #-----------------------------------------------------------------------
        # Save a checktpoint
        #-----------------------------------------------------------------------
        if (e+1) % args.checkpoint_interval == 0:
            checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
            saver.save(sess, checkpoint)
            print('Checkpoint saved:', checkpoint)

    checkpoint = '{}/final.ckpt'.format(args.name)
    saver.save(sess, checkpoint)
    print('Checkpoint saved:', checkpoint)
    