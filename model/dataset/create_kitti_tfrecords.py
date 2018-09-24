#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.misc as smi

import tensorflow as tf 

slim = tf.contrib.slim


IMAGE_HEIGHT = 1248
IMAGE_WIDTH = 384

_COLOR_MAP = {
            'road':[255,0,255],
            'background':[255,0,0]
            }

_NUM_SAMPLES = 0

 
# _EXAMPLE_FEATURE={
#     'image/height': _int64_feature(height),
#     'image/width': _int64_feature(width),
#     # 'image/format': _bytes_feature(b'png'),
#     'image/scope': _bytes_feature()
#     'image/image_raw': _bytes_feature(img_raw),
#     'image/label_raw': _bytes_feature(annotation_raw)}

# base_path = os.path.dirname(__file__)

_KEYS_TO_FEATURES = {
        'image/height': tf.FixedLenFeature([1],tf.int64),
        'image/width': tf.FixedLenFeature([1],tf.int64),
        'image/format': tf.FixedLenFeature((),tf.string, default_value='png'),
        'image/scope': tf.FixedLenFeature((),tf.string),
        'image/image_raw': tf.FixedLenFeature((),tf.string),
        'image/label_raw': tf.FixedLenFeature((),tf.string)
        }

_ITEMS_TO_HANDLERS = {
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'format': slim.tfexample_decoder.Tensor('image/format'),
        'scope': slim.tfexample_decoder.Tensor('image/scope'),
        'image': slim.tfexample_decoder.Image(
            image_key='image/image_raw',
            format_key='image/format',
            channels=3),
        'label': slim.tfexample_decoder.Image(
            image_key='image/label_raw',
            format_key='image/format',
            channels=1)
        }

def _create_example_feature(image_raw, label_raw, height, width, im_format='png', scope=''):

    return tf.train.Example(features=tf.train.Features(feature={
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/format': _bytes_feature(im_format),
                'image/scope': _bytes_feature(scope),
                'image/image_raw': _bytes_feature(image_raw),
                'image/label_raw': _bytes_feature(label_raw)}))

def _get_image_label(image_seg, color_map=_COLOR_MAP):
    """
    """
    shape = image_seg.shape
    road = np.all(image_seg == color_map['road'], axis=2)
    label = road.reshape([shape[0],shape[1],1]).astype(np.uint8)

    return label

def _create_label_png(anno_file, color_map):
    """

    Parameters
    ----------
    anno_file:
        annotation file of ground truth.
    color_map:
        instance color map.
    """
    anno = smi.imread(anno_file)
    shape = anno.shape
    road = np.all(anno == color_map['road'],axis=2)
    label = road.reshape([shape[0],shape[1]]).astype(np.uint8)

    print('label shape',label.shape)

    label_file = anno_file.replace('.png', '_label.png')
    smi.imsave(label_file, label)

    return label_file, shape


def get_dataset(file_sources, num_samples):
    # file_pattern = './kitti_segmentation.tfrecords'

    # _keys_to_features = {
    #     'image/height': tf.FixedLenFeature([1],tf.int64),
    #     'image/width': tf.FixedLenFeature([1],tf.int64),
    #     # 'image/format': tf.FixedLenFeature((),tf.string,default_value='png'),
    #     'image/image_raw': tf.FixedLenFeature((),tf.string),
    #     'image/label_raw': tf.FixedLenFeature((),tf.string)
    # }

    # _items_to_handlers = {
    #     'height': slim.tfexample_decoder.Tensor('image/height'),
    #     'width': slim.tfexample_decoder.Tensor('image/width'),
    #     # 'format': slim.tfexample_decoder.Tensor('image/format'),
    #     'image': slim.tfexample_decoder.Image('image/image_raw'),
    #     'label': slim.tfexample_decoder.Image('image/label_raw')
    # }

    base_path = os.path.dirname(__file__)
    file_sources = map(lambda fi:os.path.join(base_path, fi), file_sources)

    print(file_sources)

    decoder = slim.tfexample_decoder.TFExampleDecoder(_KEYS_TO_FEATURES, _ITEMS_TO_HANDLERS)

    # dataset = slim.dataset.Dataset(
    #     data_sources=file_pattern,
    #     reader=tf.TFRecordReader,
    #     num_samples=3,
    #     decoder=decoder,
    #     items_to_description={},
    #     )

    dataset = slim.dataset.Dataset(
            data_sources=file_sources,
            reader=tf.TFRecordReader,
            num_samples = num_samples,
            decoder=decoder,
            items_to_descriptions={},
            # num_classes=2,
            )

    # data_provider = slim.dataset_data_provider.DatasetDataProvider(
    #                     dataset,
    #                     num_readers=1,
    #                     shuffle=False)

    # height, width,image, label = provider.get(['height','width','image','label'])

    return dataset
    # print(type(image))
    # print(image.shape)


# def _image_preprocessing(image, mask, height, width, num_classes=2):



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def filename_pairs_generator(base_path, pairs_file_list):
    """
    """
    pairs_file_list = map(lambda pairs_file:os.path.join(base_path, pairs_file), pairs_file_list)
    for pairs_file in pairs_file_list:
        for line in open(pairs_file):
            line = line.strip('\n')
            image_file, label_file = line.split(" ")
            image_file = os.path.join(base_path, image_file)
            label_file = os.path.join(base_path, label_file)

            yield image_file, label_file


def create_segmentation_tfrecord(tfrecords_filename, filename_pairs_gen, split=None, scope='', im_format='png'):

    """

    Parameters
    ----------
    tfrecords_filename:
        tfrecords file name to be created.
    filename_pairs_gen:
        a generator to get [image, annotation] name pairs.
    scope:
        dataset name. 
    im_format:
        image and ground truth file format. 
    """
    # Let's collect the real images to later on compare
    # to the reconstructed ones
    # original_images = []
    num_samples = 0

    i = 0
    count = 0

    tfrecords_filename = tfrecords_filename.replace('.tfrecords', '_'+str(i)+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    new_writer = False

    for image_file, anno_file in filename_pairs_gen:
        count += 1
        if split is not None and count >= split:
            count = 0
            new_writer = True
            i += 1
        
        if new_writer:
            new_writer = False
            writer.close()
            tfrecords_filename = tfrecords_filename.replace('.tfrecords', '_'+str(i)+'.tfrecords')
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            print('writing to', tfrecords_filename)

        # img = np.array(Image.open(img_path))
        # annotation = np.array(Image.open(annotation_path))

        label_file, shape = _create_label_png(anno_file, color_map=_COLOR_MAP)

        image_raw = tf.gfile.GFile(image_file,mode='r').read()
        label_raw = tf.gfile.GFile(label_file,mode='r').read()

        # im_ori = smi.imread(image_path)
        # # im_label = smi.imread(label_path)
        # label = _get_image_label(im_label,color_map=_COLOR_MAP)
        

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = shape[0]
        width = shape[1]
        
        # Put in the original images into array
        # Just for future check for correctness
        # original_images.append((image_raw, label_raw))
        
        # image_raw = im_ori.tostring()
        # label_raw = label.tostring()
        
        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'height': _int64_feature(height),
        #     'width': _int64_feature(width),
        #     'image_raw': _bytes_feature(img_raw),
        #     'mask_raw': _bytes_feature(annotation_raw)}))

        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'image/height': _int64_feature(height),
        #     'image/width': _int64_feature(width),
        #     'image/format': _bytes_feature(b'png'),
        #     'image/image_raw': _bytes_feature(img_raw),
        #     'image/label_raw': _bytes_feature(annotation_raw)}))

        example = _create_example_feature(image_raw, label_raw, height, width, 
                        im_format=im_format, scope=scope)
        
        writer.write(example.SerializeToString())
        num_samples += 1

    writer.close()

    return num_samples



def test_gfile(filename_pairs_gen):

    for image_path, label_path in filename_pairs_gen:
        im_ori = tf.gfile.GFile(image_path,mode='r').read()
        im_label = tf.gfile.GFile(label_path,mode='r').read()

        print(im_ori.__class__)

        return 0

def test_reconstructed(tfrecords_filename, original_images):

    reconstructed_images = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        height = int(example.features.feature['image/height']
                                     .int64_list
                                     .value[0])
        
        width = int(example.features.feature['image/width']
                                    .int64_list
                                    .value[0])
        img_format = (example.features.feature['image/format']
                                      .bytes_list
                                      .value[0])

        img_scope = (example.features.feature['image/scope']
                                      .bytes_list
                                      .value[0])

        img_string = (example.features.feature['image/image_raw']
                                      .bytes_list
                                      .value[0])
        
        annotation_string = (example.features.feature['image/label_raw']
                                    .bytes_list
                                    .value[0])
        
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        
        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
        
        # Annotations don't have depth (3rd dimension)
        reconstructed_annotation = annotation_1d.reshape((height, width, -1))
        
        reconstructed_images.append((reconstructed_img, reconstructed_annotation))
        # print 

    # Let's check if the reconstructed images match
    # the original images

    # for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
        
    #     img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
    #                                                           reconstructed_pair)
    #     print(np.allclose(*img_pair_to_compare))
    #     print(np.allclose(*annotation_pair_to_compare))

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/scope': tf.FixedLenFeature([], tf.string),
        'image/image_raw': tf.FixedLenFeature([], tf.string),
        'image/label_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image/image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['image/label_raw'], tf.uint8)
    
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    annotation_shape = tf.stack([height, width, 1])
    

    print(image.shape)
    # image.set_shape(image_shape)
    # annotation.set_shape(annotation_shape)

    image = tf.image.resize_images(image, (384,1248,3))
    annotation = tf.image.resize_images(annotation, (384,1248,1))

    # image = tf.reshape(image, image_shape)
    # annotation = tf.reshape(annotation, annotation_shape)
    
    # image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    # annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    # resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                        target_height=IMAGE_HEIGHT,
    #                                        target_width=IMAGE_WIDTH)
    
    # resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
    #                                        target_height=IMAGE_HEIGHT,
    #                                        target_width=IMAGE_WIDTH)
    
    
    images, annotations = tf.train.batch( [image, annotation],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 )
    
    return  images, annotations







if __name__ == "__main__":

    scope = 'kitti'

    base_path = os.path.dirname(__file__)
    pairs_file_list = ['train_'+scope+'.txt']

    tfrecords_filename = 'train_'+scope+'_segmentation.tfrecords'

    filename_pairs_gen = filename_pairs_generator(base_path, pairs_file_list)
    # num_samples, original_images = create_segmentation_tfrecord(
    #     tfrecords_filename, filename_pairs_gen,split=None,
    #     scope=scope)
    # print("num_samples", num_samples)
    # print("create finished!!!")
    
    # test_reconstructed(tfrecords_filename, original_images)

    # test_gfile(filename_pairs_gen)

    # file_sources = os.path.join(base_path, tfrecords_filename)
    # data_provider = get_dataset([file_sources], num_samples=571)

    # height, width,image, label = data_provider.get(['height','width','image','label'])

    # with tf.Session() as sess:
    #     with tf.device("cpu:0"):
    #         height, width,image, label = sess.run([height, width,image, label])
    #         print(height, width, image.shape, label.shape)


    # count = 0
    # for image, label in data_gen:
    #     print(image, label)
        # count += 1
        # if count == 30:
        #     break

    # # tfrecords_filename = os.path.join(base_path, tfrecords_filename)
    # filename_queue = tf.train.string_input_producer(
    #     [tfrecords_filename,], num_epochs=10)

    # # # Even when reading in multiple threads, share the filename
    # # # queue.
    # image, annotation = read_and_decode(filename_queue)

    # # # The op for initializing the variables.
    # # init_op = tf.group(tf.global_variables_initializer(),
    # #                    tf.local_variables_initializer())

    # with tf.Session()  as sess:
        
    #     sess.run(init_op)
        
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
        
    #     # Let's read off 3 batches just for example
    #     for i in xrange(3):
        
    #         img, anno = sess.run([image, annotation])
    #         print(img[0, :, :, :].shape)
            
    #         print('current batch')
            
    #         # We selected the batch size of two
    #         # So we should get two image pairs in each batch
    #         # Let's make sure it is random

    #         smi.imshow(img[0, :, :, :])
    #         # smi.show()

    #         smi.imshow(anno[0, :, :, 0])
    #         # io.show()
            
    #         smi.imshow(img[1, :, :, :])
    #         # io.show()

    #         smi.imshow(anno[1, :, :, 0])
    #         # io.show()
            
        
    #     coord.request_stop()
    #     coord.join(threads)


    # get_dataset()