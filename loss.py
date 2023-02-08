import os
import tensorflow as tf
import numpy as np


def lossFrobenius(A, UI, tf_mask):
    # cost of masked, unnormalized Frobenius norm
    diff = tf.boolean_mask(tf.cast(A, tf.float32) - UI, tf_mask)
    cost = tf.reduce_sum(tf.pow(diff, 2))
    return cost
def lossCentroid(model): # cost of kmeans, unnormalized, only appears in train_allI()
    centroid_map = tf.matmul(model.I_C, model.I_assign)
    cost = tf.reduce_sum(tf.square(centroid_map - model.I))
    return cost
def lossCentroid_1(model): # cost of kmeans, unnormalized, only appears in train()
    if model.hierarchy:
        centroid_map = tf.matmul(model.I_C_1, model.I_assign_1)
        cost = tf.reduce_sum(tf.square(centroid_map - model.I_C))
        return cost
    else:
        return 0.0