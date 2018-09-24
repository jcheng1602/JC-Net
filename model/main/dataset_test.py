import tensorflow as tf

from dataset.create_kitti_tfrecords import get_dataset
from dataset import input_generator
import model_options

import scipy.misc as smi
import numpy as np

import cv2

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue



# with tf.Graph().as_default():

#     dataset = get_dataset(model_options.train_datasets, model_options.num_samples)
#     samples = input_generator.get(dataset,
#         model_options.crop_size,
#         model_options.train_batch_size,
#         num_readers=1,
#         num_threads=1,
#         is_training=True)
#     inputs_queue = prefetch_queue.prefetch_queue(samples,capacity=8)

if __name__ == "__main__":


    test_datasets = ['train_cityscapes_segmentation_0.tfrecords',]
    num_samples = 200

    crop_size = [384, 960]
    batch_size = 1


    # with tf.Session()  as sess:
    sess = tf.InteractiveSession()
    # a = tf.constant(1)
    # b = tf.constant(2)
    # c = a + b
    # d = sess.run(c)
    # print d

    s = tf.constant('campus')
    s2 = tf.constant('campus1')
    print(tf.equal(s,s2))
    result = tf.cond(tf.equal(s,s2), lambda:tf.constant(1),lambda:tf.constant(0))
    print(sess.run(result))
    # tf.cond(tf. )
    # tf.cond(a<b, lambda:tf.)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    dataset = get_dataset(test_datasets, num_samples)
    samples = input_generator.get(dataset,
        crop_size,
        batch_size,
        num_readers=3,
        num_threads=3,
        is_training=True)


    inputs_queue = prefetch_queue.prefetch_queue(samples,capacity=8*model_options.train_batch_size)

    samples_ = inputs_queue.dequeue()

    print(samples_)

    # images = samples_['image']

    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())
    
    # sess.run(init_op)   
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        sample_ = sess.run(samples_)
        image_ = sample_['image']
        label_ = sample_['label']
        smi.imsave('temp/{}_im_t.png'.format(i),image_[0,:,:,:])
        smi.imsave('temp/{}_la_t.png'.format(i),label_[0,:,:])
        # print(sess.run(samples_).__class__)

        print(image_.max(),image_.shape)
        cv2.imshow('train_im',image_[0,:,:,:].astype(np.uint8))
        k = cv2.waitKey(0)
        if k==27:
            cv2.destroyWindow('train_im')

    # print images_

    # # smi.imshow(images_[0,:,:,:])

    coord.request_stop()
    coord.join(threads)

