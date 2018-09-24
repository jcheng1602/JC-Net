
import os
import time

import tensorflow as tf
# from deeplab import common
# from deeplab import model
# from deeplab.datasets import segmentation_dataset
# from deeplab.utils import input_generator
# from deeplab.utils import train_utils

from model import build_training_model, segment_images
from decoder.segmentation import compute_loss
import train_utils
from dataset.create_kitti_tfrecords import get_dataset
from dataset import input_generator
import model_options


slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue


# model_options = ModelOptions()

train_logdir = './log/'+ time.strftime("%Y%m%d_%H%M%S",time.localtime())

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    #dataset = 
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():

        with tf.device('cpu:0'):
            dataset = get_dataset(model_options.train_datasets, model_options.num_samples)
            samples = input_generator.get(dataset,
                model_options.crop_size,
                model_options.train_batch_size,
                num_readers=1,
                num_threads=1,
                is_training=True)
            inputs_queue = prefetch_queue.prefetch_queue(samples,capacity=8)

        # with tf.device('gpu:0'):
        global_step = tf.train.get_or_create_global_step()
        images, labels, pred_logits, pred_labels = build_training_model(inputs_queue, model_options)
        compute_loss(pred_logits, labels, model_options)
        pred_images = segment_images(images, pred_labels)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # Add summaries for model variables.
        # for model_var in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # # Add summaries for losses.
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        #     summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # with tf.device('gpu:0'):
        learning_rate = train_utils.get_model_learning_rate(
            model_options.learning_policy, model_options.base_learning_rate,
            model_options.learning_rate_decay_step, model_options.learning_rate_decay_factor,
            model_options.training_number_of_steps, model_options.learning_power,
            model_options.slow_start_step, model_options.slow_start_learning_rate)

        # optimizer = tf.train.MomentumOptimizer(learning_rate, model_options.momentum)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.00001)

        # for variable in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))

        # with tf.device('gpu:0'):
        total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

        # train_op = slim.learning.create_train_op(total_loss, optimizer)


        grads_and_vars = optimizer.compute_gradients(total_loss)
        grads, tvars = zip(*grads_and_vars)
        clip_norm = 1.
        clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)
        grads_and_vars = zip(clipped_grads, tvars)
        grad_updates = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
            # acc = tf.identity(acc)


        slim.summaries.add_image_summary(images,'image')
        slim.summaries.add_image_summary(pred_images,'pred_image')

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies([update_ops]):
        #     train_tensor = optimizer.apply_gradients(total_loss, global_step=global_step)

        # summaries.add(tf.summary.scalar('total_loss',total_loss))
        slim.summaries.add_scalar_summary(train_tensor,'total_loss',print_summary=True)
        # slim.summaries.add_scalar_summary(acc,'accuracy',print_summary=True)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/accuracy': slim.metrics.streaming_accuracy(pred_labels, labels),
            'eval/precision': slim.metrics.streaming_precision(pred_labels, labels),
        })
        for metric_name, metric_value in names_to_values.iteritems():
            # op = tf.summary.scalar(metric_name, metric_value)
            # op = tf.Print(op, [metric_value], metric_name)
            # summaries.add(op)
            with tf.control_dependencies(names_to_updates.values()):
                slim.summaries.add_scalar_summary(
                    metric_value, metric_name, print_summary=True)
        


        #Create the summary ops such that they also print out to std output:
        # summary_ops = []
        # for metric_name, metric_value in names_to_values.iteritems():
        #     # op = tf.summary.scalar(metric_name, metric_value)
        #     # op = tf.Print(op, [metric_value], metric_name)
        #     # summaries.add(op)
        #     with tf.control_dependencies(names_to_updates.values()):
        #         slim.summaries.add_scalar_summary(
        #             metric_value, metric_name, print_summary=True)


        # summaries |= set(
        #     tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        # summary_op = tf.summary.merge(list(summaries))
        summary_op = tf.summary.merge_all()

        # print('summary_op',summary_op)

        # Soft placement allows placing on CPU ops without GPU implementation.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        slim.learning.train(
            train_tensor,
            logdir=train_logdir,
            log_every_n_steps=model_options.log_steps,
            number_of_steps=model_options.training_number_of_steps,
            session_config=session_config,
            # init_fn=train_utils.get_model_init_fn(
            #     train_logdir,
            #     model_options.tf_initial_checkpoint,
            #     model_options.initialize_last_layer,
            #     model_options.last_layers,
            #     ignore_missing_vars=True),
            summary_op=summary_op,
            save_summaries_secs=model_options.save_summaries_secs,
            save_interval_secs=model_options.save_interval_secs)   


