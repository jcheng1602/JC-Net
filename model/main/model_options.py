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
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
# import collections

# import tensorflow as tf
import os
# Flags for input preprocessing.

# tf_initial_checkpoint = './log/20180419_221132/model.ckpt-95767'
tf_initial_checkpoint ='/home/kevin/Downloads/deeplabv3_cityscapes_train/model.ckpt'
initialize_last_layer = False
last_layers = []


min_resize_value = None # Desired size of the smaller image side.
max_resize_value = None # Maximum allowed size of the larger image side.

resize_factor = None # Resized dimensions are multiple of factor plus one.

# Model dependent flags.
logits_kernel_size = 1 # The kernel size for the convolutional kernel that generates logits.

# When using 'xception_65', we set atrous_rates = [6, 12, 18] (output stride 16)
# and decoder_output_stride = 4.
image_pyramid =  None # Input scales for multi-scale feature extraction.
add_image_level_feature = False # Add image level feature.

aspp_with_batch_norm = True  # Use batch norm parameters for ASPP or not.
aspp_with_separable_conv = True # Use separable convolution for ASPP or not.

multi_grid  = None # Employ a hierarchy of atrous rates for ResNet.

depth_multiplier =1.0 # Multiplier for the depth (number of channels) for all
                      # convolution ops used in MobileNet.

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
decoder_output_stride = 4 # The ratio of input to output spatial resolution when 
                             # employing decoder to refine segmentation results.
decoder_use_separable_conv =  True # Employ separable convolution for decoder or not.
merge_method = 'max' # ['max', 'avg'], Scheme to merge multi scale features.

outputs_to_num_classes = 2
atrous_rates = [6,12,18]
output_stride = 16

# crop_size = [384,1248]
# crop_size = [256,960]
crop_size = [384,960]
# crop_size = [512,1280]
train_batch_size = 2


## Settings for training strategy.
learning_policy = 'poly' # ['poly', 'step'], 'Learning rate policy for training.'
# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
base_learning_rate = .0001 # 'The base learning rate for model training.'
learning_rate_decay_factor = 0.1 # 'The rate to decay the base learning rate.'
learning_rate_decay_step = 2000 # 'Decay the base learning rate at a fixed step.'
learning_power = 0.9 # 'The power value used in the poly learning policy.'
training_number_of_steps = 200000 # 'The number of steps used for training'
momentum = 0.9 # 'The momentum value to use'
slow_start_step = 0 # Training model with small learning rate for few steps
slow_start_learning_rate = 1e-4 # Learning rate employed during slow start.


# train_logdir = './log/'+ time.strftime("%Y%m%d_%H%M%S",time.localtime()
log_steps = 10
save_summaries_secs = 60   
save_interval_secs = 600


# train_datasets = ['train_campus_segmentation.tfrecords','train_kitti_segmentation.tfrecords',
#     'train_cityscapes_segmentation_0.tfrecords', 'train_cityscapes_segmentation_1.tfrecords',
#     'train_cityscapes_segmentation_2.tfrecords','train_cityscapes_segmentation_3.tfrecords',
#     'train_cityscapes_segmentation_4.tfrecords','train_cityscapes_segmentation_5.tfrecords',]
# num_samples = 3399

train_datasets = ['train_cityscapes_segmentation_0.tfrecords','train_cityscapes_segmentation_1.tfrecords','train_cityscapes_segmentation_2.tfrecords',
'train_cityscapes_segmentation_3.tfrecords','train_cityscapes_segmentation_4.tfrecords','train_cityscapes_segmentation_5.tfrecords',]
num_samples = 289

# train_datasets = ['train_cityscapes_segmentation_1.tfrecords',]
# num_samples = 100

val_datasets = []

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


# class ModelOptions(
#     collections.namedtuple('ModelOptions', [
#         'outputs_to_num_classes',
#         'crop_size',
#         'atrous_rates',
#         'output_stride',
#         'merge_method',
#         'add_image_level_feature',
#         'aspp_with_batch_norm',
#         'aspp_with_separable_conv',
#         'multi_grid',
#         'decoder_output_stride',
#         'decoder_use_separable_conv',
#         'logits_kernel_size',
#         'train_batch_size'
#     ])):
#   """Immutable class to hold model options."""

#   __slots__ = ()

#   def __new__(cls,
#               outputs_to_num_classes=outputs_to_num_classes,
#               crop_size=crop_size,
#               atrous_rates=atrous_rates,
#               output_stride=output_stride):
#     """Constructor to set default values.

#     Args:
#       outputs_to_num_classes: A dictionary from output type to the number of
#         classes. For example, for the task of semantic segmentation with 21
#         semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
#       crop_size: A tuple [crop_height, crop_width].
#       atrous_rates: A list of atrous convolution rates for ASPP.
#       output_stride: The ratio of input to output spatial resolution.

#     Returns:
#       A new ModelOptions instance.
#     """
#     return super(ModelOptions, cls).__new__(
#         cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
#         merge_method, add_image_level_feature,
#         aspp_with_batch_norm, aspp_with_separable_conv,
#         multi_grid, decoder_output_stride,
#         decoder_use_separable_conv, logits_kernel_size,
#         train_batch_size
#         )
