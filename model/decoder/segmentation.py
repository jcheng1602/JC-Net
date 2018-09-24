import tensorflow as tf

from encoder import feature_extractor


slim = tf.contrib.slim

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'

# DECODER_END_POINTS = 'entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'
DECODER_END_POINTS = 'entry_flow/block2/unit_1/xception_module/separable_conv2'

def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):

  outputs = slim.separable_conv2d(
      inputs,
      None,
      3,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def _extract_features_with_aspp(features, 
                                model_options,
                                weight_decay=0.0001,
                                reuse=None,
                                is_training=False,
                                fine_tune_batch_norm=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  # features, end_points = feature_extractor.extract_features(
  #     images,
  #     output_stride=model_options.output_stride,
  #     multi_grid=model_options.multi_grid,
  #     model_variant=model_options.model_variant,
  #     weight_decay=weight_decay,
  #     reuse=reuse,
  #     is_training=is_training,
  #     fine_tune_batch_norm=fine_tune_batch_norm)  

  # if not model_options.aspp_with_batch_norm:
  #   return features, end_points
  # else:

  with tf.variable_scope('aspp'):
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        # 'is_training': is_training, 
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
        # 'scale': False,
    }

    # with slim.arg_scope(
    #     [slim.conv2d, slim.separable_conv2d],
    #     weights_regularizer=slim.l2_regularizer(weight_decay),
    #     activation_fn=tf.nn.relu,  
    #     normalizer_fn=slim.batch_norm,
    #     padding='SAME',
    #     stride=1,
    #     reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,      
        normalizer_fn=slim.batch_norm,          # changed
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        depth = 256
        branch_logits = []

        # if model_options.add_image_level_feature:
        #   pool_height = scale_dimension(model_options.crop_size[0],
        #                                 1. / model_options.output_stride)
        #   pool_width = scale_dimension(model_options.crop_size[1],
        #                                1. / model_options.output_stride)
        #   # pool_size = tf.shape(images)[1:3]
        #   image_feature = slim.avg_pool2d(
        #       features, [pool_height, pool_width], [pool_height, pool_width],
        #       padding='VALID')
        #   image_feature = slim.conv2d(
        #       image_feature, depth, 1, scope=_IMAGE_POOLING_SCOPE)
        #   image_feature = tf.image.resize_bilinear(
        #       image_feature, [pool_height, pool_width], align_corners=True)
        #   image_feature.set_shape([None, pool_height, pool_width, depth])
        #   branch_logits.append(image_feature)

        # if model_options.add_image_level_feature:
        #   pool_size = tf.shape(features)[1:3]
        #   image_feature = slim.avg_pool2d(
        #       features, pool_size, pool_size,
        #       padding='VALID')
        #   image_feature = slim.conv2d(
        #       image_feature, depth, 1, scope=_IMAGE_POOLING_SCOPE)
        #   image_feature = tf.image.resize_bilinear(
        #       image_feature, pool_size, align_corners=True)
        #   # image_feature.set_shape([None, pool_height, pool_width, depth])
        #   branch_logits.append(image_feature)

        # Employ a 1x1 convolution.
        branch_logits.append(slim.conv2d(features, depth, 1,
                                         scope=_ASPP_SCOPE + str(0)))

        if model_options.atrous_rates:
          # Employ 3x3 convolutions with different atrous rates.
          for i, rate in enumerate(model_options.atrous_rates, 1):
            scope = _ASPP_SCOPE + str(i)
            if model_options.aspp_with_separable_conv:
              aspp_features = _split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            branch_logits.append(aspp_features)

        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = slim.conv2d(
            concat_logits, depth, 1, scope=_CONCAT_PROJECTION_SCOPE)
        # concat_logits = slim.dropout(
        #     concat_logits,
        #     keep_prob=0.9,
        #     is_training=is_training,
        #     scope=_CONCAT_PROJECTION_SCOPE + '_dropout')

        # concat_logits = dropout_selu(
        #       concat_logits, 
        #       rate=0.9, 
        #       training=is_training, 
        #       name=_CONCAT_PROJECTION_SCOPE + '_dropout')

        return concat_logits

def _refine_by_decoder(features,   # concat_logits
                      end_points,
                      # decoder_height,
                      # decoder_width,
                      model_options,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
  """Adds the decoder to obtain sharper segmentation results.

  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the network to the corresponding
      activation.
    decoder_height: The height of decoder feature maps.
    decoder_width: The width of decoder feature maps.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].
  """
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      # 'is_training': is_training,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  # decoder_height = scale_dimension(model_options.crop_size[0],
  #                                     1. / model_options.decoder_output_stride)
  # decoder_width = scale_dimension(model_options.crop_size[1],
  #                                     1. / model_options.decoder_output_stride)

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      # activation_fn=tf.nn.relu,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      # normalizer_fn=None,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features]):
        print(end_points)
        middle_feature_name = '{}/{}'.format('xception_fast', DECODER_END_POINTS)
        middle_feature = slim.conv2d(
                                  end_points[middle_feature_name],
                                  48,
                                  1,
                                  scope='feature_projection')
        encoder_shape = tf.shape(middle_feature)[1:3]
        final_feature = tf.image.resize_bilinear(
              features, encoder_shape, align_corners=True)
        decoder_features_list = [middle_feature, final_feature]

        decoder_depth = 256
        if decoder_use_separable_conv:
          decoder_features = _split_separable_conv2d(
              tf.concat(decoder_features_list, 3),
              filters=decoder_depth,
              rate=1,
              weight_decay=weight_decay,
              scope='decoder_conv0')
          decoder_features = _split_separable_conv2d(
              decoder_features,
              filters=decoder_depth,
              rate=1,
              weight_decay=weight_decay,
              scope='decoder_conv1')
        else:
          num_convs = 2
          decoder_features = slim.repeat(
              tf.concat(decoder_features_list, 3),
              num_convs,
              slim.conv2d,
              decoder_depth,
              3,
              scope='decoder_conv' + str(i))

        return decoder_features

        # feature_list = DECODER_END_POINTS
        # if feature_list is None:
        #   tf.logging.info('Not found any decoder end points.')
        #   return features
        # else:
        #   decoder_features = features
        #   for i, name in enumerate(feature_list):
        #     decoder_features_list = [decoder_features]
        #     feature_name = '{}/{}'.format('xception_fast', name)
        #     decoder_features_list.append(
        #         slim.conv2d(
        #             end_points[feature_name],
        #             48,
        #             1,
        #             scope='feature_projection' + str(i)))
        #     # Resize to decoder_height/decoder_width.
        #     for j, feature in enumerate(decoder_features_list):
        #       decoder_features_list[j] = tf.image.resize_bilinear(
        #           feature, [decoder_height, decoder_width], align_corners=True)
        #       decoder_features_list[j].set_shape(
        #           [None, decoder_height, decoder_width, None])
        #     decoder_depth = 256
        #     if decoder_use_separable_conv:
        #       decoder_features = _split_separable_conv2d(
        #           tf.concat(decoder_features_list, 3),
        #           filters=decoder_depth,
        #           rate=1,
        #           weight_decay=weight_decay,
        #           scope='decoder_conv0')
        #       decoder_features = _split_separable_conv2d(
        #           decoder_features,
        #           filters=decoder_depth,
        #           rate=1,
        #           weight_decay=weight_decay,
        #           scope='decoder_conv1')
        #     else:
        #       num_convs = 2
        #       decoder_features = slim.repeat(
        #           tf.concat(decoder_features_list, 3),
        #           num_convs,
        #           slim.conv2d,
        #           decoder_depth,
        #           3,
        #           scope='decoder_conv' + str(i))

          # return decoder_features

# def _refine_by_decoder(features,   # concat_logits
#                       end_points,
#                       # decoder_height,
#                       # decoder_width,
#                       model_options,
#                       decoder_use_separable_conv=False,
#                       model_variant=None,
#                       weight_decay=0.0001,
#                       reuse=None,
#                       is_training=False,
#                       fine_tune_batch_norm=False):
#   """Adds the decoder to obtain sharper segmentation results.

#   Args:
#     features: A tensor of size [batch, features_height, features_width,
#       features_channels].
#     end_points: A dictionary from components of the network to the corresponding
#       activation.
#     decoder_height: The height of decoder feature maps.
#     decoder_width: The width of decoder feature maps.
#     decoder_use_separable_conv: Employ separable convolution for decoder or not.
#     model_variant: Model variant for feature extraction.
#     weight_decay: The weight decay for model variables.
#     reuse: Reuse the model variables or not.
#     is_training: Is training or not.
#     fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

#   Returns:
#     Decoder output with size [batch, decoder_height, decoder_width,
#       decoder_channels].
#   """
#   batch_norm_params = {
#       # 'is_training': is_training and fine_tune_batch_norm,
#       'is_training': is_training,
#       'decay': 0.9997,
#       'epsilon': 1e-5,
#       'scale': True,
#   }

#   decoder_height = scale_dimension(model_options.crop_size[0],
#                                       1. / model_options.decoder_output_stride)
#   decoder_width = scale_dimension(model_options.crop_size[1],
#                                       1. / model_options.decoder_output_stride)

#   with slim.arg_scope(
#       [slim.conv2d, slim.separable_conv2d],
#       weights_regularizer=slim.l2_regularizer(weight_decay),
#       # activation_fn=tf.nn.relu,
#       activation_fn=tf.nn.relu,
#       # normalizer_fn=slim.batch_norm,
#       normalizer_fn=None,
#       padding='SAME',
#       stride=1,
#       reuse=reuse):
#     with slim.arg_scope([slim.batch_norm], **batch_norm_params):
#       with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features]):
#         feature_list = DECODER_END_POINTS
#         if feature_list is None:
#           tf.logging.info('Not found any decoder end points.')
#           return features
#         else:
#           decoder_features = features
#           for i, name in enumerate(feature_list):
#             decoder_features_list = [decoder_features]
#             feature_name = '{}/{}'.format('xception_fast', name)
#             decoder_features_list.append(
#                 slim.conv2d(
#                     end_points[feature_name],
#                     48,
#                     1,
#                     scope='feature_projection' + str(i)))
#             # Resize to decoder_height/decoder_width.
#             for j, feature in enumerate(decoder_features_list):
#               decoder_features_list[j] = tf.image.resize_bilinear(
#                   feature, [decoder_height, decoder_width], align_corners=True)
#               decoder_features_list[j].set_shape(
#                   [None, decoder_height, decoder_width, None])
#             decoder_depth = 256
#             if decoder_use_separable_conv:
#               decoder_features = _split_separable_conv2d(
#                   tf.concat(decoder_features_list, 3),
#                   filters=decoder_depth,
#                   rate=1,
#                   weight_decay=weight_decay,
#                   scope='decoder_conv0')
#               decoder_features = _split_separable_conv2d(
#                   decoder_features,
#                   filters=decoder_depth,
#                   rate=1,
#                   weight_decay=weight_decay,
#                   scope='decoder_conv1')
#             else:
#               num_convs = 2
#               decoder_features = slim.repeat(
#                   tf.concat(decoder_features_list, 3),
#                   num_convs,
#                   slim.conv2d,
#                   decoder_depth,
#                   3,
#                   scope='decoder_conv' + str(i))

#           return decoder_features

def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=None,
                       scope_suffix=''):
  """Gets the logits from each model's branch.

  The underlying model is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].

  Raises:
    ValueError: Upon invalid input kernel_size value.
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in _extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
    with tf.variable_scope(_LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                features,
                num_classes,
                kernel_size=kernel_size,
                rate=rate,
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits)


def predict_logits(images, model_options, weight_decay=0.0001, is_training=False, reuse=None, fine_tune_batch_norm=False):
  
  features, end_points = feature_extractor.extract_features(
    images,
    output_stride=model_options.output_stride, 
    multi_grid=model_options.multi_grid, 
    depth_multiplier=1.0, 
    # final_endpoint=None, 
    # model_variant=None, 
    weight_decay=weight_decay, 
    reuse=reuse, 
    is_training=is_training, 
    fine_tune_batch_norm=fine_tune_batch_norm, 
    regularize_depthwise=True, 
    preprocess_images=True, 
    # num_classes=None, 
    # global_pool=False
    )

  # print "features", features.get_shape()

  aspp_logits = _extract_features_with_aspp(
    features, 
    model_options, 
    weight_decay=0.0001, 
    reuse=reuse, 
    is_training=is_training, 
    fine_tune_batch_norm=fine_tune_batch_norm)

  # print "aspp_logits", aspp_logits.get_shape()

  refine_logits = _refine_by_decoder(
    aspp_logits, 
    end_points, 
    model_options,
    decoder_use_separable_conv=model_options.decoder_use_separable_conv,
    # model_variant=None,
    weight_decay=0.0001,
    reuse=reuse,
    is_training=is_training,
    fine_tune_batch_norm=fine_tune_batch_norm)

  # print "refine_logits", refine_logits.get_shape()

  pred_logits = _get_branch_logits(
    refine_logits, 
    num_classes=model_options.outputs_to_num_classes,
    atrous_rates=model_options.atrous_rates,
    aspp_with_batch_norm=model_options.aspp_with_batch_norm,
    kernel_size=model_options.logits_kernel_size,
    weight_decay=0.0001,
    reuse=reuse,
    scope_suffix='')

  with tf.variable_scope('logits'):
    pred_logits = tf.image.resize_bilinear(
          pred_logits,
          tf.shape(images)[1:3],
          align_corners=True)

  softmax = tf.nn.softmax(pred_logits, axis=3)
  pred_labels = tf.argmax(softmax, axis=-1, output_type=tf.int32)

  return pred_logits, pred_labels


def compute_loss(pred_logits, labels, model_options):
  """
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  with tf.variable_scope('loss'):
    labels = tf.reshape(labels,shape=[-1])
    pred_logits = tf.reshape(pred_logits, shape=[-1, model_options.outputs_to_num_classes])

    one_hot_labels = slim.one_hot_encoding(
      labels, model_options.outputs_to_num_classes, on_value=1.0, off_value=0.0)

    tf.losses.softmax_cross_entropy(
        one_hot_labels, 
        pred_logits)


def get_eval_ops(pred_labels, labels):
  """
  """
  pass
