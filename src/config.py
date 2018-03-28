# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mamiruzz"
__date__ = "$Feb 4, 2018 7:36:59 PM$"

import tensorflow as tf
import dataset

def GetNewWeights(_shape=None, _name=None):
    with tf.name_scope(name=_name):
        return tf.Variable(tf.truncated_normal(_shape, stddev=0.05))

def GetNewBiases(_length=0, _name=None):
    with tf.name_scope(name=_name):
        return tf.Variable(tf.constant(0.05, shape=[_length]))


def GetNewConvLayer(_input=None, # The previous layer.
                    _num_input_channels=0, # Num. channels in prev. layer.
                    _filter_size=0, # Width and height of each filter.
                    _num_filters=0, # Number of filters.
                    _use_pooling=True, _name=None):  # Use 2x2 max-pooling.
    with tf.name_scope(name=_name):
        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [_filter_size, _filter_size, _num_input_channels, _num_filters]

        # Create new weights aka. filters with the given shape.
        weights = GetNewWeights(_shape=shape, _name="W")

        # Create new biases, one for each filter.
        biases = GetNewBiases(_length=_num_filters, _name="B")

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=_input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if _use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def GetFlattenLayer(_layer=None, _name=None):
    with tf.name_scope(name=_name):
        # Get the shape of the input layer.
        layer_shape = _layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(_layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def GetNewFCLayer(_input=None, # The previous layer.
                  _num_inputs=0, # Num. inputs from prev. layer.
                  _num_outputs=0, # Num. outputs.
                  _use_relu=True, _name=None): # Use Rectified Linear Unit (ReLU)?
    with tf.name_scope(name=_name):

        # Create new weights and biases.
        weights = GetNewWeights(_shape=[_num_inputs, _num_outputs], _name="W_fc")
        biases = GetNewBiases(_length=_num_outputs, _name="B_fc")

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(_input, weights) + biases

        # Use ReLU?
        if _use_relu:
            layer = tf.nn.relu(layer)

    return layer

def PrintProgress(_epoch=0, _feed_dict_train=None, _feed_dict_validate=None, _val_loss=None, _accuracy=None):
    # Calculate the accuracy on the training-set.
    with tf.Session() as session:
        acc = session.run(_accuracy, feed_dict=_feed_dict_train)
        val_acc = session.run(_accuracy, feed_dict=_feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(_epoch + 1, acc, val_acc, val_loss))

import time
from datetime import timedelta

# Counter for total number of iterations performed so far.
total_iterations = 0
def Optimize(_num_iterations=0, _train_batch_size=0, 
             _img_size_flat=0, _optimizer=None, _accuracy=None, 
             _cost=None, _num_examples=None, _early_stopping=None, 
             _model_name=None, _train_path=None, _img_size=0, _classes=None, _validation_size=None, _x=None, _y_true=None):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + _num_iterations):
                       
        data = dataset.read_train_sets(_train_path, _img_size, _classes, _validation_size)
        #test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(_train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(_train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(_train_batch_size, _img_size_flat)
        x_valid_batch = x_valid_batch.reshape(_train_batch_size, _img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        num_classes = len(_classes)
#        x = tf.placeholder(tf.float32, shape=[None, _img_size_flat], name='x')
#        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        feed_dict_train = {_x: x_batch,
                           _y_true: y_true_batch}
        
        feed_dict_validate = {_x: x_valid_batch,
                              _y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session = tf.Session()
        session.run(_optimizer, feed_dict=feed_dict_train)
        
        ##############
        # saving model
        ##############
#        
#        saver = tf.train.Saver()
#        #saver.save(session, 'C:\\Users\\mamiruzz\\Downloads\\Netbeans\\Python\\MulticlassClassification\\src\\'+modelName) 
#        saver.save(session, './' + _model_name) 
#        saver = tf.train.Saver()
#        save_path = saver.save(session, _model_name+'.meta')
#        print("Model saved in file: %s" % save_path)
#        with tf.Session() as sess:
#            new_saver = tf.train.import_meta_graph(_model_name + '.meta')
#            new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(_num_examples / _train_batch_size) == 0: 
            val_loss = session.run(_cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(_num_examples / _train_batch_size))

            PrintProgress(_epoch=epoch, _feed_dict_train=feed_dict_train, 
                          _feed_dict_validate=feed_dict_validate, _val_loss=val_loss, 
                          _accuracy=_accuracy)

            if _early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == _early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += _num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

import matplotlib.pyplot as plt
    
def PlotImages(_images=None, _cls_true=None, _cls_pred=None, _img_size=0, _num_channels=0):
    
    if len(_images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(_images)), min(len(_images), 9))
        
        
    _images, _cls_true  = zip(*[(_images[i], _cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(_images[i].reshape(_img_size, _img_size, _num_channels))

        # Show true and predicted classes.
        if _cls_pred is None:
            xlabel = "True: {0}".format(_cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(_cls_true[i], _cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

    
def PlotExampleErrors(_cls_pred=None, _correct=None):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (_correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = _cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    
    # Plot the first 9 images.
    PlotImages(_images=images[0:9],
               _cls_true=cls_true[0:9],
               _cls_pred=cls_pred[0:9],
               _img_size=img_size, 
               _num_channels=num_channels)


def PlotConfusionMatrix(_cls_pred=None, _cls_true=None, _num_classes=0):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    #cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=_cls_true,
                          y_pred=_cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(_num_classes)
    plt.xticks(tick_marks, range(_num_classes))
    plt.yticks(tick_marks, range(_num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def PrintValidationAccuracy(show_example_errors=False,
                            show_confusion_matrix=False, _num_test=0, _batch_size=0, 
                            _images=None, _labels=None, _y_pred_cls=None, 
                            _cls_true=None, _classes=0):

    # Number of images in the test-set.
    #num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=_num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < _num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + _batch_size, _num_test)

#        # Get the images from the test-set between index i and j.
#        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
#        
#
#        # Get the associated labels.
#        labels = data.valid.labels[i:j, :]

    # Create a feed-dict with these images and labels.
    feed_dict = {x: _images,
        y_true: _labels}

    # Calculate the predicted class using TensorFlow.
    cls_pred[i:j] = session.run(_y_pred_cls, feed_dict=feed_dict)

    # Set the start-index for the next batch to the
    # end-index of the current batch.
    i = j

    cls_true = _cls_true
    cls_pred = np.array([_classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / _num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, _num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        PlotExampleErrors(_cls_pred=cls_pred, _correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        PlotConfusionMatrix(cls_pred=cls_pred)

import numpy as np
def SamplePrediction(_test_im=None, _x=None,_img_size_flat=0,_y_true=None, _y_pred_cls=None):
    
    feed_dict_test = {
        _x: _test_im.reshape(1, _img_size_flat),
        _y_true: np.array([[1, 0]])
    }
    with tf.Session() as sess:
        test_pred = sess.run(_y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]

def plotSingleImage(_image=None, _img_size=0, _num_channels=0, _title=None):
    plt.title(_title)
    plt.imshow(_image.reshape(_img_size, _img_size, _num_channels),
               interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    print ("Config is called")