import os
import tensorflow as tf
import numpy as np
import pandas as pd
from loss import *
#######################################################################################
# Loading the matrix to be decomposed
def read_data_ml100k():
    data_dir = '/movielens_100k/'
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names, engine='python')
    return data
def read_data_ml1m():
    data_dir = '/movielens_1m/'
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), '\::', names=names, engine='python')
    return data



def PCA_1stVec_split(M, row_indices):
    M_reduced_T = tf.boolean_mask(tf.transpose(M), row_indices)
    normalized_data = normalize(M_reduced_T)
    eigen_values, eigen_vectors = tf.linalg.eigh(tf.matmul(tf.transpose(normalized_data), normalized_data))
    PCA_Vec = eigen_vectors[:,-1]
    dist = tf.einsum("ab,b->a", normalized_data, PCA_Vec)
    return dist
def normalize(M):
    X = tf.identity(M)
    X -=tf.reduce_mean(M, axis=0)
    return X
def isprime(num):
    for n in range(2,int(num**0.5)+1):
        if num%n==0:
            return False
    return True



def print_info(step, A, model, tf_mask_train, tf_mask_validation, train_frac, total_data):
    train_loss = lossFrobenius(A, model(), tf_mask_train)
    validation_loss = lossFrobenius(A, model(), tf_mask_validation)
    centroid_val_loss = lossFrobenius(A, model(pred_with_centroid=True), tf_mask_validation)
    tf.print("Step %2d: train_loss=%2.5f, val_loss=%2.5f, val_loss_with_centroid=%2.5f"
             % (step, train_loss/train_frac/total_data,
                validation_loss/(1-train_frac)/total_data,
                centroid_val_loss/(1-train_frac)/total_data))
    # tf.print("Centroid distribution:", tf.reduce_sum(model.I_assign, 1))
    if model.base_model == "NeuNMF":
        tf.print("Frac:", tf.math.sigmoid(model.frac))
    return None