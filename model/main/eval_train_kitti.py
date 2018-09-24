#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time 

import tensorflow as tf

import numpy as np 
from glob import glob

import cv2

from devkit_road import evaluateRoad

import matplotlib.pyplot as plt

'''测试算法在kitti上的几个典型指标
'''


def load_graph(frozen_graph_filename, prefix='', return_elements=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name=prefix)
    
    return graph

def build_graph():
    """Build deep learning graph.
    """
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    # frozen_model = '/home/kevin/Codes/DeepNet/log/20180419_221132/frozen_model.pb'
    # frozen_model = '/home/kevin/Downloads/deeplabv3_cityscapes_train/frozen_inference_graph.pb'
    # frozen_model = '/home/kevin/Codes/EnvNet/RUNS/used3/frozen_model.pb'
    frozen_model = '/home/kevin/Codes/DeepNet/log/20180713_114748/frozen_model.pb'
    graph = load_graph(frozen_model)

    for op in graph.get_operations():
        print(op.name)

    ## model_envnet/frozen_model.pb
    image_pl = graph.get_tensor_by_name('ImagePlaceholder:0')
    pred_seg = graph.get_tensor_by_name('SemanticPredictions:0')

    ## model_deeplab/frozen_inference_graph.pb
    # image_pl = graph.get_tensor_by_name('ImageTensor:0')
    # pred_seg = graph.get_tensor_by_name('SemanticPredictions:0')

    # ## model_deepnet/frozen_model.pb
    # image_pl = graph.get_tensor_by_name('ImagePlaceholder:0')
    # pred_seg = graph.get_tensor_by_name('SemanticPredictions:0')

    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(graph=graph,config=config)

    return image_pl, pred_seg, sess


def detection_process(images, image_pl, pred_seg, sess):
    """
    image shape [b,h,w,c]
    """
    feed_dict = {image_pl: images}
#    (np_pred_boxes, np_pred_confidences, np_pred_seg) = sess.run([pred_boxes,
#                                                        pred_confidences,
#                                                        pred_seg],
#                                                        feed_dict=feed_dict)  # < 50m
    np_pred_seg = sess.run(pred_seg, feed_dict=feed_dict)  # < 50m                                                        
                                                    
    # np_pred_seg = np.reshape(np_pred_seg, (images.shape[0], -1, 2))
    # np_pred_seg = np_pred_seg[:,:,1]

    seges = np_pred_seg

    return seges


def main(result_file_dir,train_file_dir):

    image_pl, pred_seg, sess = build_graph()

    # train_file_dir = '/home/kevin/Codes/EnvNet/DATA/data_road/training/image_2'
    # result_file_dir = '/home/kevin/Codes/DeepNet/result_envnet'

    # print('start')
    fn_search = '*.png'
    train_fileList = glob(os.path.join(train_file_dir+'/image_2',fn_search))

    time1 = time.time()
    for train_file in train_fileList:

        file_key = train_file.split('/')[-1].split('.')[0]
        # get tags
        tags = file_key.split('_')
        ts_tag = tags[-1]
        dataset_tag = tags[0]
        class_tag = 'road'

        result_tag = dataset_tag + '_' + class_tag + '_' + ts_tag

        # train_im = cv2.imread(train_file)
        train_im = (plt.imread(train_file)*255.).astype(np.uint8)
        # train_im = cv2.resize(train_im, (1024,256))
        print(train_im.max())

        seges = detection_process(np.expand_dims(train_im,0), image_pl, pred_seg, sess)
        print('sege',seges.shape)

        # seges = (seges == 0).astype(np.uint8)

        result_file = os.path.join(result_file_dir,result_tag+'.png')
        cv2.imwrite(result_file,(seges[0,:,:,1]*255.).astype(np.uint8))
        # print('finish')

        # cv2.imshow('train_im',train_im)
        # k = cv2.waitKey(0)
        # if k==27:
        #     cv2.destroyWindow('train_im')
        # # plt.show()

        # break

    time2 = time.time()   
    print('write time:',time2-time1)




if __name__ == "__main__":

    train_file_dir = '/home/kevin/Codes/EnvNet/DATA/data_road/training'
    result_file_dir = '/home/kevin/Codes/DeepNet/result1/20180713_114748'

    main(result_file_dir,train_file_dir)

    time1 = time.time()
    evaluateRoad.main(result_file_dir,train_file_dir)
    time2 = time.time()
    print('cost time:',time2-time1)