import os
import tensorflow as tf
from decoder.segmentation import predict_logits

#from common import ModelOptions
import model_options

slim = tf.contrib.slim

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'


# model_options = ModelOptions()   
        # outputs_to_num_classes=2,
        # crop_size=[384,1248],
        # atrous_rates=[6,12,18],
        # output_stride=16)


def build_training_model(inputs_queue, model_options):

    with tf.variable_scope('inputs'):
        samples = inputs_queue.dequeue()
        images = samples['image']
        labels = samples['label']
        
    pred_logits, pred_labels = predict_logits(
        images,
        model_options,
        weight_decay=0.0001,
        is_training=True,
        reuse=None,
        fine_tune_batch_norm=False)
    # print(pred_labels.get_shape(), labels)
    # acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels,labels),tf.int32))
    # acc = tf.identity(acc)

    # acc, ops = tf.metrics.accuracy(labels, pred_labels)
    # print(acc,ops)

    # print(images)

    return images, labels, pred_logits, pred_labels

def segment_images(images, pred_labels):

    # pred_images = tf.identity(images[0,:,:,:])
    pred_labels = tf.cast(tf.identity(pred_labels),tf.uint8)
    # road_id = tf.equal(pred_labels, 1)
    # pred_images[road_id,:3] = [255.,0.,255.]
    # pred_images = pred_images * tf.expand_dims(tf.cast(pred_labels,tf.float32),3)
    pred_images = tf.expand_dims(pred_labels*tf.constant(255,dtype=tf.uint8),3)

    return pred_images



if __name__  == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    with tf.Graph().as_default() as graph:
        image_pl = tf.placeholder(dtype=tf.float32,shape=(1,384,1248,3))
        pred_logits = seg_logits.predict_logits(
            image_pl, 
            model_options, 
            weight_decay=0.0001, 
            is_training=False, 
            reuse=None, 
            fine_tune_batch_norm=False)


    print pred_logits.op.name
