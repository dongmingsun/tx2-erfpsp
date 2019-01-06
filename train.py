import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import model
import random
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/new', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')

#Training arguments
flags.DEFINE_integer('num_classes', 27, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 20, 'The batch size used for validation.')
flags.DEFINE_integer('image_height',480, "The input height of the images.")
flags.DEFINE_integer('image_width', 640, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')



FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size #Can be larger than train_batch as no need to backpropagate gradients.


#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

#Use median frequency balancing or not

#Visualization and where to save images
save_images = FLAGS.save_images
photo_dir = os.path.join(FLAGS.logdir, "images")

#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.jpg')])
annotation_files = sorted([os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if file.endswith('.png')])

image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.jpg')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if file.endswith('.png')])



#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================
#Median frequency balancing class_weights

class_weights=np.array([8.258104 , 8.779362 , 8.389586 , 8.687806 , 6.865416 , 8.554129 ,
       8.775371 , 8.778263 , 8.624385 , 8.506881 , 8.7678175, 3.601893 ,
       7.067162 , 8.567163 , 8.265365 , 7.9689765, 8.283569 , 4.4337897,
       8.59206  , 8.806689 , 8.796311 , 2.7741733, 4.0381026, 8.071181 ,
       7.9188976, 8.299413 , 0.       ], dtype=np.float32)
#============= TRAINING =================
def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy. 

    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want. 
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.

    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------

    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.

    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss

def decode(a,b):
    a = tf.read_file(a)
    a = tf.image.decode_image(a, channels=3)
    a = tf.image.convert_image_dtype(a, dtype=tf.float32)
    #a = tf.image.random_brightness(a, max_delta=0.3)
    #a.set_shape(shape=(360, 480, 3))
    b = tf.read_file(b)
    b = tf.image.decode_image(b)
    #b.set_shape(shape=(360, 480,1))
    bb,bs,_=tf.image.sample_distorted_bounding_box(tf.shape(b),bounding_boxes=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4]),min_object_covered=0.5)
    a=tf.slice(a,bb,bs)
    b=tf.slice(b,bb,bs)
    a=tf.image.resize_images(a, [image_height,image_width],method=1)
    b=tf.image.resize_images(b, [image_height,image_width],method=1)
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width,1))
    return a,b










	
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        tdataset = tf.data.Dataset.from_tensor_slices((images,annotations))
        tdataset = tdataset.map(decode)
        tdataset = tdataset.shuffle(300).batch(batch_size).repeat(num_epochs)
        titerator = tdataset.make_initializable_iterator()
        images,annotations = titerator.get_next()		


        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        logits, probabilities= model.train(images,numclasses=num_classes, shape=[image_height,image_width], l2=weight_decay,reuse=None)
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)

        #Actually compute the loss
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)

        total_loss = tf.losses.get_total_loss()

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update)

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val, _ = sess.run([train_op, global_step, accuracy, mean_IOU, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f', global_step_count, total_loss, time_elapsed, accuracy_val, mean_IOU_val)

            return total_loss, accuracy_val, mean_IOU_val

        #================VALIDATION BRANCH========================
        #Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        vdataset = tf.data.Dataset.from_tensor_slices((images_val,annotations_val))
        vdataset = vdataset.map(decode)
        vdataset = vdataset.shuffle(300).batch(eval_batch_size).repeat(num_epochs*3)
        viterator = vdataset.make_initializable_iterator()
        images_val,annotations_val = viterator.get_next()		
        logits_val, probabilities_val= model.train(images_val,numclasses=num_classes, shape=[image_height,image_width], l2=weight_decay,reuse=True)
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)
        annotations_ohe_vals =tf.split(annotations_ohe_val,num_or_size_splits=27,axis=-1)
        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        predictions_vals = tf.one_hot(predictions_val, num_classes, axis=-1)
        predictions_valss = tf.split(predictions_vals,num_or_size_splits=27,axis=-1)
        accuracy_val, accuracy_val_update = tf.contrib.metrics.streaming_accuracy(predictions_val, annotations_val)
        mean_IOU_val, mean_IOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val, labels=annotations_val, num_classes=num_classes)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update)

        #Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val, shape=[-1, image_height, image_width, 1])

        def eval_step(sess, metrics_op):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, accuracy_value, mean_IOU_value,av,pv = sess.run([metrics_op, accuracy_val, mean_IOU_val,annotations_ohe_vals,predictions_valss])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    (%.2f sec/step)', accuracy_value, mean_IOU_value, time_elapsed)

            return accuracy_value, mean_IOU_value,av,pv

        #=====================================================

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val, max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=None)
        # Run the managed session
        with sv.managed_session() as sess:
            sess.run([viterator.initializer,titerator.initializer])
            for step in range(int(num_steps_per_epoch * num_epochs)):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)

                #Log the summaries every 10 steps or every end of epoch, which ever lower.
                if step % min(num_steps_per_epoch, 10) == 0:
                    loss, training_accuracy, training_mean_IOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

                    #Check the validation data only at every third of an epoch
                    if step % (num_steps_per_epoch / 3) == 0:
                        ors=np.zeros((num_classes-1,), dtype=np.float32)
                        ans=np.zeros((num_classes-1,), dtype=np.float32)
                        for i in range(len(image_val_files) // eval_batch_size):
                            validation_accuracy, validation_mean_IOU,st,nt = eval_step(sess, metrics_op_val)
                            for j in range(num_classes-1):
                                ors[j]=ors[j]+np.logical_or(st[j],nt[j]).sum()
                                ans[j]=ans[j]+np.logical_and(st[j],nt[j]).sum()
                        vali_class=ans/(ors+0.0000001)
                        print(vali_class)
                        print(np.mean(vali_class))
                        sv.saver.save(sess, './log/others', global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, training_accuracy,training_mean_IOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

            #We log the final training loss
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Final Validation Accuracy: %s', validation_accuracy)
            logging.info('Final Validation Mean IOU: %s', validation_mean_IOU)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()
