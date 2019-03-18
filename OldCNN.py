from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from montecarlo import ConfigList, MagList, TempList
from testmonte import TestConfigList, TestMagList, TestTempList

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model Function for Convolutional Neural Network"""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [2,2],
        padding = "same",
        activation = tf.nn.relu)


    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 64,
        kernel_size = [2,2],
        padding = "same",
        activation = tf.nn.relu)

    # Dense layer

    pool2_flat = tf.reshape(conv2, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 2048, activation = tf.nn.softmax)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    
    train_data = ConfigList/np.float32(1) #Training Data
    train_labels = MagList.astype(np.int32) #Training Labels
    eval_data = TestConfigList/np.float32(1) #Test Data
    eval_labels = TestMagList.astype(np.int32) #Test Labels

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/2dising_convnet_model_large")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)    



    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # train one step and display the probabilties
    classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()
