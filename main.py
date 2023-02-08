import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
import pandas as pd
from utils import *
from CEL import *
from loss import *
from ops import *

"""==================== read data ===================="""
seed = 0
np.random.seed(seed)
train_frac = 0.8
data = read_data_ml100k()
N_U = 943
N_I = 1682
shape = (N_U, N_I)
data = data.sample(frac=1)
length = len(data.index)
train_length = int(train_frac*length)

A_orig, mask_0, mask_train, mask_val = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
_ = 0
for line in data.itertuples():
    user_index, item_index = int(line[1] - 1), int(line[2] - 1)
    score = int(line[3])
    A_orig[user_index, item_index] = score
    mask_0[user_index, item_index] = 1
    if _>train_length:
        mask_val[user_index, item_index] = 1
    else:
        mask_train[user_index, item_index] = 1
    _ += 1
A_orig_df = pd.DataFrame(A_orig)
total_data = np.sum(mask_0)
# Boolean mask for computing cost only on valid (not missing) entries
tf_mask_train = mask_train
tf_mask_validation = mask_val
A = tf.constant(A_orig_df.values)

"""==================== hyper-parameters ===================="""
base_model = "NMF" # "NMF"; "MLP"; or "NeuNMF" which combines MLP and NMF
growth_mode = "data_number"
balance = "nonzero"
split_mode = "PCA"
threshold = "zero" # zero/median/data
centroid = 50 # max M_q
initial_centroid = 1 # M_0
centroid_1 = 1
rank = 64
reduced_rank = 16
steps = range(6000) # Number of steps
start_train_all_I = 800
c1 = 1 # Norm penalty
stop_norm_step = 400000
c2 = 50 # lambda_reg for personalization
c2_1 = 500
lr = 0.0001/train_frac # Learning rate
lrcr = 1.0 # Learning rate ratio for centroid vs user embbeding
lrpr = 1.0 # Learning rate personalization ratio
lr_params = 0.001/train_length # Learning rate for parameters in MLP
reduce_lr_step = 10000
reassign_interval = 40 # t_1 (how frequent reassign)
growth_interval = 10 # t_2
initial_frac_logit = -1.0 # Determine the initial weight of the MLP prediction

"""==================== model setup ===================="""
model = CEL(initialization="abs",
            base_model=base_model,
            initial_cluster=initial_centroid,
            rank=rank,
            reduced_rank=reduced_rank,
            N_I=N_I,
            N_U=N_U,
            centroid=centroid,
            centroid_1=centroid_1,
            initial_frac_logit=initial_frac_logit)
opt_U = tf.keras.optimizers.SGD(learning_rate=lr)
opt_I_C = tf.keras.optimizers.SGD(learning_rate=lrcr*lr)
opt_I = tf.keras.optimizers.SGD(learning_rate=lrpr*lr)
opt_params = tf.keras.optimizers.Adam(learning_rate=lr_params)

"""==================== define the embedding optimization ===================="""
def train(step, model, A, tf_mask, normalization = "I"):
    model.I.assign(tf.matmul(model.I_C, model.I_assign))
    with tf.GradientTape() as t:
        current_loss = lossFrobenius(A, model(pred_with_centroid=True), tf_mask) # centroid prediction loss
        current_loss += 0.5*(np.sign(stop_norm_step-step)+1) * c1 * (tf.reduce_sum(tf.square(model.U)) + tf.reduce_sum(tf.square(model.I_C)))
        if model.hierarchy:
            current_loss += c2_1 * lossCentroid_1(model)
    dU, dI_C = t.gradient(current_loss, [model.U, model.I_C])
    if "I" in normalization: # Gradient averaging
        distribution = tf.reduce_sum(model.I_assign, 1)
        distribution = tf.maximum(distribution, tf.ones_like(distribution))  # maintain well condition of division
        dI_C = tf.matmul(dI_C, tf.linalg.diag(1/distribution)) # normalization, MOST important step
    #### sub-grad
    reduced_ratio = int(step / reduce_lr_step)+1.0
    dU, dI_C = 1.0 / reduced_ratio * dU, 1.0 / reduced_ratio * dI_C

    opt_U.apply_gradients(zip([dU], [model.U]))
    opt_I_C.apply_gradients(zip([dI_C], [model.I_C]))
    ##################################################################
    if model.base_model == "MLP" or model.base_model == "NeuNMF":
        with tf.GradientTape() as t2:
            current_loss = lossFrobenius(A, model(pred_with_centroid=True), tf_mask)  # centroid prediction loss
        dParams = t2.gradient(current_loss, [model.wu1, model.wu2, model.bu1, model.bu2,
                                             model.wi1, model.wi2, model.bi1, model.bi2])
        opt_params.apply_gradients(zip(dParams, [model.wu1, model.wu2, model.bu1, model.bu2,
                                                 model.wi1, model.wi2, model.bi1, model.bi2]))
        if model.base_model == "NeuNMF":
            with tf.GradientTape() as t3:
                current_loss = lossFrobenius(A, model(pred_with_centroid=True), tf_mask)
            dFrac = t3.gradient(current_loss, [model.frac])
            model.frac.assign_sub(tf.cast(lr_params,tf.float32)*dFrac[0])
        pass
    # Clipping operation. This ensures non-negative
    model.U.assign(tf.maximum(tf.zeros_like(model.U), model.U))
    model.I_C.assign(tf.maximum(tf.zeros_like(model.I_C), model.I_C))
    if model.hierarchy:
        filter_C = np.zeros([centroid])
        for i in range(model.N_C):
            filter_C[i] += 1
        filter_C = tf.cast(tf.linalg.diag(filter_C), tf.float32)
        filtered_distribution_1 = tf.reduce_sum(tf.matmul(model.I_assign_1,filter_C), 1)
        filtered_distribution_1 = tf.maximum(filtered_distribution_1, tf.ones_like(filtered_distribution_1))  # maintain well condition of division
        filtered_I_assign_1_T = tf.transpose(tf.matmul(model.I_assign_1,filter_C))
        new_I_C_1 = tf.matmul(tf.matmul(model.I_C, filtered_I_assign_1_T),
                                     tf.linalg.diag(1 / filtered_distribution_1))
        model.I_C_1.assign(new_I_C_1)
def train_allI(step, model, A, tf_mask,
               update_U=True, update_centroid=True, update_MLP=False): # Personalization
    with tf.GradientTape() as t:
        current_loss = lossFrobenius(A, model(), tf_mask) + c2 * lossCentroid(model)
        current_loss += 0.5*(np.sign(stop_norm_step-step)+1) * c1 * (tf.reduce_sum(tf.square(model.U)) + tf.reduce_sum(tf.square(model.I)))
    dU, dI = t.gradient(current_loss, [model.U, model.I])
    reduced_ratio = int(step/reduce_lr_step)+1.0
    dU, dI = 1.0/reduced_ratio*dU, 1.0/reduced_ratio*dI
    opt_I.apply_gradients(zip([dI], [model.I]))
    model.I.assign(tf.maximum(tf.zeros_like(model.I), model.I))
    if update_U:
        opt_U.apply_gradients(zip([dU], [model.U]))
        model.U.assign(tf.maximum(tf.zeros_like(model.U), model.U))
    if update_centroid:
        distribution = tf.reduce_sum(model.I_assign, 1)
        distribution = tf.maximum(distribution, tf.ones_like(distribution))
        model.I_C.assign(tf.matmul(tf.matmul(model.I, tf.transpose(model.I_assign)), tf.linalg.diag(1 / distribution)))
    if update_MLP:
        raise ValueError("TBI.")


"""==================== main function ===================="""
for step in steps:
    if model.hierarchy:
        reassign_vanilla_1(model, centroid, centroid_1)
    if step % growth_interval == 0 and step < start_train_all_I and model.N_C < centroid:
        growth(model, A, tf_mask_train, centroid, threshold,
               mode=growth_mode,split_mode=split_mode)
        distribution = tf.reduce_sum(model.I_assign, 1)
    if step < start_train_all_I:
        train(step, model, A, tf_mask_train)
    else: # Personalization
        train_allI(step, model, A, tf_mask_train)

    if (step+1) % reassign_interval == 0 and step < start_train_all_I:
        n = int((step+1) / reassign_interval)
        if isprime(n):
            changes = reassign(model, A, tf_mask_train, centroid)
    elif step >= start_train_all_I:
        reassign_vanilla(model, centroid, N_I)
    if step % 10 == 0 or step <= 10:
        print_info(step, A, model, tf_mask_train, tf_mask_validation, train_frac, total_data)
