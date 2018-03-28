# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mamiruzz"
__date__ = "$Feb 4, 2018 7:35:14 PM$"

#libraries
import cv2
import dataset
from datetime import timedelta
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3 #best to have the filter size as odd number
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['dogs', 'cats']
#classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_classes = len(classes)

# batch size
batch_size = 32

# validation split
validation_size = .16

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping



train_path = 'training_data'
test_path = 'testing_data'





checkpoint_dir = "models/"

modelName = "Amir_model"

dropout = 0.25 # Dropout, probability to drop a unit
is_training = True

############
#load data
############
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size, classes)
#test_ids test file name array--can be used to write the result.
#print(test_images)
#print(test_ids)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

print('test labels:\n')
#print(data.test.labels)

session = tf.Session() #move this to model
# Get some random images and their labels from the train set.

images, cls_true  = data.train.images, data.train.cls
#cls_true is class name array

#print(images)
#print(cls_true)

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#print(x)
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
#print(x_image)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
#print(y_true)

#y_true_cls = tf.argmax(y_true, dimension=1)--updating deprecated function
y_true_cls = tf.argmax(y_true, axis=1)
#print(y_true_cls)


#def GetNewConvLayer(_input=None, # The previous layer.
#                    _num_input_channels=0, # Num. channels in prev. layer.
#                    _filter_size=0, # Width and height of each filter.
#                    _num_filters=0, # Number of filters.
#                    _use_pooling=True):  # Use 2x2 max-pooling.

import config


layer_conv1, weights_conv1 = \
    config.GetNewConvLayer(_input=x_image,
                           _num_input_channels=num_channels,
                           _filter_size=filter_size1,
                           _num_filters=num_filters1,
                           _use_pooling=True, _name="Conv1")

    
#print(layer_conv1)



layer_conv2, weights_conv2 = \
    config.GetNewConvLayer(_input=layer_conv1,
                           _num_input_channels=num_filters1,
                           _filter_size=filter_size2,
                           _num_filters=num_filters2,
                           _use_pooling=True, _name="conv2")
                   


layer_conv3, weights_conv3 = \
    config.GetNewConvLayer(_input=layer_conv2,
                           _num_input_channels=num_filters2,
                           _filter_size=filter_size3,
                           _num_filters=num_filters3,
                           _use_pooling=True, _name="conv3")
                   
   
layer_flat, num_features = config.GetFlattenLayer(layer_conv3, _name="FlattenLayer1")
#print(layer_flat)
#print(num_features)




layer_fc1 = config.GetNewFCLayer(_input=layer_flat,
                                 _num_inputs=num_features,
                                 _num_outputs=fc_size,
                                 _use_relu=True, _name="FC1")
                         
layer_fc2 = config.GetNewFCLayer(_input=layer_fc1,
                                 _num_inputs=fc_size,
                                 _num_outputs=num_classes,
                                 _use_relu=False, _name="FC2")
# Apply Dropout (if is_training is False, dropout is not applied)
layer_fc2 = tf.layers.dropout(layer_fc2, rate=dropout, training=is_training)

y_pred = tf.nn.softmax(layer_fc2)
#print(y_pred)
##y_pred_cls = tf.argmax(y_pred, dimension=1)--updating deprecated parameter
y_pred_cls = tf.argmax(y_pred, axis=1)
#print(y_pred_cls)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ' + str(accuracy))
# Initialize the variables.
#init = tf.global_variables_initializer()

# Initialize the variables.
session.run(tf.global_variables_initializer())


train_batch_size = batch_size

# Counter for total number of iterations performed so far.
total_iterations = 0

#x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
#x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)


#PrintProgress(_epoch=0, _feed_dict_train=None, _feed_dict_validate=None, _val_loss=None, _accuracy=None)


##############start#########################

def optimize(num_iterations):
    # Initialize the variables.
    session.run(tf.global_variables_initializer())
    
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
            y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
            y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        
        ##############
        # saving model
        ##############
        saver = tf.train.Saver()
        #saver.save(session, 'C:\\Users\\mamiruzz\\Downloads\\Netbeans\\Python\\MulticlassClassification\\src\\'+modelName) 
        saver.save(session, './' + modelName) 

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples / batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))
            
            #print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            config.PrintProgress(_epoch=epoch, _feed_dict_train=feed_dict_train, 
                                 _feed_dict_validate=feed_dict_validate, _val_loss=val_loss, 
                                 _accuracy=accuracy)
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))
    
##############end##################################


num_examples = data.train.num_examples

num_test = len(data.valid.images)
#print(num_test)
#print(data.valid.images)

# Get the images from the test-set between index i and j.
#images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
#        
#
## Get the associated labels.
#labels = data.valid.labels[i:j, :]
cls_true = np.array(data.valid.cls)

import os
if os.path.exists('{}.meta'.format(modelName)):
    #model.load(modelName)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelName + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print('model loaded!')
else:
    print("Creating model: " + "{}.meta".format(modelName))
    total_iterations = 0

    optimize(num_iterations=2)
#    config.Optimize(_num_iterations=1, _train_batch_size=train_batch_size, 
#             _img_size_flat=img_size_flat, _optimizer=optimizer, _accuracy=accuracy, 
#             _cost=cost, _num_examples=num_examples, _early_stopping=early_stopping, 
#             _model_name=modelName, _train_path=train_path, _img_size=img_size, 
#             _classes=classes, _validation_size=validation_size, _x=x, _y_true=y_true)
    #config.PrintValidationAccuracy(show_example_errors=True, show_confusion_matrix=True)

#Optimize(_num_iterations=0, _x_batch=None, _y_true_batch=None,
#             _x_valid_batch=None, _y_valid_batch=None, _train_batch_size=0, 
#             _img_size_flat=0, _optimizer=None, _accuracy=None, 
#             _cost=None, _num_examples=None, _early_stopping=None, 
#             _model_name=None)


test_image = cv2.imread('C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\MultiClassTensorflow\\src\\testing_data\\dogs\\dog.1000.jpg')
test_image = cv2.resize(test_image, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_dog = plt.imshow(test_image.reshape(img_size, img_size, num_channels))


#test_image = cv2.resize(test_image, (img_size, img_size), interpolation = cv2.INTER_AREA)

def sample_prediction(test_im):
    
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[1, 0]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]

title = "Predicted class for test_image: {}".format(sample_prediction(test_image))
print(title)


#image1 = test_images[0]
#config.plotSingleImage(_image=image1, _img_size=img_size, _num_channels=num_channels)


config.plotSingleImage(_image=test_image, _img_size=img_size, 
                       _num_channels=num_channels, _title=title)


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:,:, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.
    
    image = image.reshape(img_size_flat)

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0,:,:, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()




plot_conv_weights(weights=weights_conv1)

plot_conv_layer(layer=layer_conv1, image=test_image)


print("done")
#write tensorlog
writer = tf.summary.FileWriter('C:\\temp\\tensorflow_logs\\', session.graph)
