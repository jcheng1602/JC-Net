# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentation data."""

import tensorflow as tf
import preprocess_utils

slim = tf.contrib.slim

dataset_data_provider = slim.dataset_data_provider


# def _get_data(data_provider, dataset_split):
#   """Gets data from data provider.

#   Args:
#     data_provider: An object of slim.data_provider.
#     dataset_split: Dataset split.

#   Returns:
#     image: Image Tensor.
#     label: Label Tensor storing segmentation annotations.
#     image_name: Image name.
#     height: Image height.
#     width: Image width.

#   Raises:
#     ValueError: Failed to find label.
#   """
#   if common.LABELS_CLASS not in data_provider.list_items():
#     raise ValueError('Failed to find labels.')

#   image, height, width = data_provider.get(
#       [common.IMAGE, common.HEIGHT, common.WIDTH])

#   # Some datasets do not contain image_name.
#   if common.IMAGE_NAME in data_provider.list_items():
#     image_name, = data_provider.get([common.IMAGE_NAME])
#   else:
#     image_name = tf.constant('')

#   label = None
#   if dataset_split != common.TEST_SET:
#     label, = data_provider.get([common.LABELS_CLASS])

#   return image, label, image_name, height, width


# def _get_data(data_provider):

_KEYS_TO_FEATURES = {
        'image/height': tf.FixedLenFeature([1],tf.int64),
        'image/width': tf.FixedLenFeature([1],tf.int64),
        'image/format': tf.FixedLenFeature((),tf.string,default_value='png'),
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


def get_dataset(file_sources, num_samples):
    # file_pattern = './kitti_segmentation.tfrecords'

    # _keys_to_features = {
    #     'image/height': tf.FixedLenFeature([1],tf.int64),
    #     'image/width': tf.FixedLenFeature([1],tf.int64),
    #     'image/format': tf.FixedLenFeature((),tf.string,default_value='png'),
    #     'image/image_raw': tf.FixedLenFeature((),tf.string),
    #     'image/label_raw': tf.FixedLenFeature((),tf.string)
    # }

    # _items_to_handlers = {
    #     'height': slim.tfexample_decoder.Tensor('image/height'),
    #     'width': slim.tfexample_decoder.Tensor('image/width'),
    #     'format': slim.tfexample_decoder.Tensor('image/format'),
    #     'image': slim.tfexample_decoder.Image('image/image_raw'),
    #     'label': slim.tfexample_decoder.Image('image/label_raw')
    # }


    decoder = slim.tfexample_decoder.TFExampleDecoder(_KEYS_TO_FEATURES, _ITEMS_TO_HANDLERS)

    dataset = slim.dataset.Dataset(
            data_sources=file_sources,
            reader=tf.TFRecordReader,
            num_samples = num_samples,
            decoder=decoder,
            items_to_descriptions = {},
            # num_classes=2)
            )

    return dataset
    # data_provider = slim.dataset_data_provider.DatasetDataProvider(
    #                     dataset,
    #                     num_readers=1,
    #                     shuffle=False)

    # height, width,image, label = provider.get(['height','width','image','label'])
    # return data_provider

def _random_crop_image_and_label(image, label, crop_height, crop_width):

    label = tf.cast(label, tf.float32)

    image_and_label = tf.concat([image, label], axis=3)
    image_and_label = tf.squeeze(image_and_label)
    croped_image_label = tf.random_crop(image_and_label, [crop_height,crop_width,4])
    crop_image, crop_label = croped_image_label[:,:,:3],croped_image_label[:,:,-1]

    crop_label = tf.cast(crop_label, tf.int32)
    # crop_label = tf.reshape(crop_label, [crop_height,crop_width,1])
    crop_image.set_shape([crop_height,crop_width,3])
    crop_label.set_shape([crop_height,crop_width])

    return crop_image, crop_label



def preprocess_image_and_label(image,
                               label,
                               scope,
                               crop_height,
                               crop_width,
                               is_training=True):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  # if model_variant is None:
  #   tf.logging.warning('Default mean-subtraction is performed. Please specify '
  #                      'a model_variant. See feature_extractor.network_map for '
  #                      'supported model variants.')

  # Keep reference to original image.
  # original_image = image

  # processed_image = tf.cast(image, tf.float32)

  # image = tf.cond(tf.equal(scope,tf.constant('kitti')),
  #                           lambda: tf.image.resize_bilinear(image, [384,1248])
  #                           lambda: image)
  # label = tf.cond(tf.equal(scope,tf.constant('kitti')),
  #                           lambda: tf.image.resize_bilinear(image, [384,1248])
  #                           lambda: image)

  # image = 

  # print('image shape', image.get_shape())
  # print('label shape', label.get_shape())

  image = tf.cast(image, tf.float32)
  if label is not None:
    label = tf.cast(label, tf.int32)

  # image_and_label = tf.concat([image, label], axis=2)



  image = tf.expand_dims(image, 0)
  label = tf.expand_dims(label, 0)


  crop_image = tf.cond(tf.equal(scope,tf.constant('campus')),
                  lambda: image[:,280:792,4:-4,:],
                  lambda: image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('campus')),
                  lambda: label[:,280:792,4:-4,:],
                  lambda: label)

  new_dim = tf.to_int32([384, 960])
  crop_image = tf.cond(tf.equal(scope,tf.constant('campus')),
                  lambda: tf.image.resize_bilinear(crop_image, new_dim, align_corners=True),
                  lambda: crop_image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('campus')),
                  lambda: tf.image.resize_nearest_neighbor(crop_label,new_dim,align_corners=True),
                  lambda: crop_label)

  crop_image = tf.cond(tf.equal(scope,tf.constant('cityscapes')),
                  # lambda: crop_image[:,156:975,8:-8,:],
                  lambda: crop_image[:,:800,8:-8,:],
                  lambda: crop_image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('cityscapes')),
                  # lambda: crop_label[:,156:975,8:-8,:],
                  lambda: crop_label[:,:800,8:-8,:],
                  lambda: crop_label)

  new_dim_city = tf.to_int32([384, 960])
  crop_image = tf.cond(tf.equal(scope,tf.constant('cityscapes')),
                  lambda: tf.image.resize_bilinear(crop_image, new_dim_city, align_corners=True),
                  lambda: crop_image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('cityscapes')),
                  lambda: tf.image.resize_nearest_neighbor(crop_label,new_dim_city,align_corners=True),
                  lambda: crop_label)

  crop_image = tf.cond(tf.equal(scope,tf.constant('kitti')),
                  lambda: crop_image[:,:365,8:-8,:],
                  lambda: crop_image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('kitti')),
                  lambda: crop_label[:,:365,8:-8,:],
                  lambda: crop_label)

  new_dim_kitti = tf.to_int32([384, 960])
  crop_image = tf.cond(tf.equal(scope,tf.constant('kitti')),
                  lambda: tf.image.resize_bilinear(crop_image, new_dim_kitti, align_corners=True),
                  lambda: crop_image)
  crop_label = tf.cond(tf.equal(scope,tf.constant('kitti')),
                  lambda: tf.image.resize_nearest_neighbor(crop_label,new_dim,align_corners=True),
                  lambda: crop_label)  


  crop_image = tf.squeeze(crop_image)
  crop_label = tf.squeeze(crop_label)
  crop_image.set_shape([384, 960, 3])
  crop_label.set_shape([384, 960])
  # crop_image, crop_label = _random_crop_image_and_label(image, label, crop_height, crop_width)

  return crop_image, crop_label


def get(dataset,
        crop_size,
        batch_size,
        # min_resize_value=None,
        # max_resize_value=None,
        # resize_factor=None,
        # min_scale_factor=1.,
        # max_scale_factor=1.,
        # scale_factor_step_size=0,
        num_readers=1,
        num_threads=1,
        is_training=True,
        ):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    crop_size: Image crop size [height, width].
    batch_size: Batch size.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.

  Raises:
    ValueError: dataset_split is None, failed to find labels, or label shape
      is not valid.
  """
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      # num_epochs=None,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
  # image, label, image_name, height, width = _get_data(data_provider,
  #                                                     dataset_split)
  image, label, height, width, scope = data_provider.get(
        ['image', 'label', 'height', 'width', 'scope'])

  image, label = preprocess_image_and_label(
      image,
      label,
      scope,
      crop_height=crop_size[0],  
      crop_width=crop_size[1],
      is_training=is_training)

  print('image',image.get_shape())
  print('label', label.get_shape())

  # image = tf.image.resize_images(image,crop_size)
  # label = tf.image.resize_images(label,crop_size)

  sample = {
      'image': image,
      # 'label': label,
      # 'scope': scope,
      # 'height': height,
      # 'width': width
  }

  if label is not None:
    sample['label'] = label

  if not is_training:
    # Original image is only used during visualization.
    # sample[common.ORIGINAL_IMAGE] = original_image,
    num_threads = 1




  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)
