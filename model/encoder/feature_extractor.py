import functools
import tensorflow as tf

from xception_modify import xception_fast, xception_arg_scope
# from nets.mobilenet import mobilenet as mobilenet_lib
# from nets.mobilenet import mobilenet_v2


slim = tf.contrib.slim

# Names for end point features.
# DECODER_END_POINTS = 'decoder_end_points'
DECODER_END_POINTS = 'entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'

# # A dictionary from network name to a map of end point features.
# networks_to_feature_maps = {
#     'mobilenet_v2': {
#         # The provided checkpoint does not include decoder module.
#         DECODER_END_POINTS: None,
#     },
#     'xception_65': {
#         DECODER_END_POINTS: [
#             'entry_flow/block2/unit_1/xception_module/'
#             'separable_conv2_pointwise',
#         ],
#     }
# }

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
# name_scope = {
#     'mobilenet_v2': 'MobilenetV2',
#     'xception_65': 'xception_65',
# }

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0


# _PREPROCESS_FN = {
#     'mobilenet_v2': _preprocess_zero_mean_unit_range,
#     'xception_65': _preprocess_zero_mean_unit_range,
# }


def mean_pixel(model_variant=None):
  """Gets mean pixel value.

  This function returns different mean pixel value, depending on the input
  model_variant which adopts different preprocessing functions. We currently
  handle the following preprocessing functions:
  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.

  Args:
    model_variant: Model variant (string) for feature extraction. For
      backwards compatibility, model_variant=None returns _MEAN_RGB.

  Returns:
    Mean pixel value.
  """
  if model_variant is None:
    return _MEAN_RGB
  else:
    return [127.5, 127.5, 127.5]


def extract_features(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     # final_endpoint=None,
                     # model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     num_classes=None,
                     global_pool=False):
  """Extracts features by the parituclar model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    output_stride: The ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the network up to.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    preprocess_images: Performs preprocessing on images or not. Defaults to
      True. Set to False if preprocessing will be done by other functions. We
      supprot two types of preprocessing: (1) Mean pixel substraction and (2)
      Pixel values normalization to be [-1, 1].
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.

  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Unrecognized model variant.
  """

  if preprocess_images:
    images = _preprocess_zero_mean_unit_range(images)

  arg_scope = xception_arg_scope(
      weight_decay=weight_decay,
      batch_norm_decay=0.9997,
      batch_norm_epsilon=1e-3,
      batch_norm_scale=True,
      regularize_depthwise=regularize_depthwise,
      use_batch_norm=True)   # 

  with slim.arg_scope(arg_scope):
    features, end_points = xception_fast(
          inputs=images,
          num_classes=num_classes,
          is_training=(is_training and fine_tune_batch_norm),
          # is_training=is_training,
          global_pool=global_pool,
          output_stride=output_stride,
          regularize_depthwise=regularize_depthwise,
          multi_grid=multi_grid,
          reuse=reuse,
          scope='xception_fast')


  return features, end_points

