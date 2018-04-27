import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random
import pandas as pd

filter_size1 = 3 
num_filters1 = 32
filter_size2 = 3
num_filters2 = 32

filter_size3 = 3
num_filters3 = 64
    
fc_size = 128

num_channels = 3

img_size = 50

img_size_flat = img_size * img_size * num_channels

img_shape = (img_size, img_size)

classes = ['xx-24', '25-xx']
num_classes = len(classes)

batch_size = 16

validation_size = 0.1

early_stopping = None

train_path = #path to images for taining
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
    	             filter=weights,
    	             strides=[1, 1, 1, 1],
    	             padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#3 convolutional layers with the following properties
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
               num_input_channels=num_channels,
               filter_size=filter_size1,
               num_filters=num_filters1,
               use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
               num_input_channels=num_filters1,
               filter_size=filter_size2,
               num_filters=num_filters2,
               use_pooling=True)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
               num_input_channels=num_filters2,
               filter_size=filter_size3,
               num_filters=num_filters3,
               use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size,
                     use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                     num_inputs=fc_size,
                     num_outputs=num_classes,
                     use_relu=False)

#adam optimizer with a learning rate of 10^(-4) is used
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer()) #new
#session.run(tf.initialize_all_variables()) #old
train_batch_size = batch_size

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    best_val_loss = float("inf")
    for i in range(total_iterations,total_iterations + num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        feed_dict_train = {x: x_batch,y_true: y_true_batch}      
        feed_dict_validate = {x: x_valid_batch,y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_train)        
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
    total_iterations += num_iterations


optimize(num_iterations=9800)

train_batch_size = 682
x_valid_batch, y_valid_batch,y_id, valid_cls_batch = data.valid.next_batch(train_batch_size)
x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
feed_dict_validate = {x: x_valid_batch,y_true: y_valid_batch}
val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

test_batch_size = len(test_images)
x_test_images = test_images
x_test_images = x_test_images.reshape(test_batch_size, img_size_flat)
feed_dict_test = {x:x_test_images}
test_pred = session.run(y_pred_cls,feed_dict_test)

test_pred_df = pd.DataFrame(data=test_pred, columns=["cnn_age"])
test_df = pd.DataFrame(data=[t.strip(".jpg") for t in test_ids], columns=["userid"])
test_final_df = pd.concat([test_df,test_pred_df], axis=1)
print(test_final_df) 