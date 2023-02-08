import os
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from utils import *
from loss import *

"""==================== cluster optimization ===================="""

def reassign(model, A, tf_mask, centroid, balancing="nonzero"):
    for i in range(model.N_C):
        centroid_slice_repeat = tf.repeat(tf.expand_dims(model.I_C[:,i], axis=1), model.N_I, axis=1) # (rank, N_I)
        cost_slice = tf.square(tf.cast(A,tf.float32) - model(temp_I=centroid_slice_repeat))
        cost_slice = tf.einsum("ui,ui->i",cost_slice,tf_mask)
        if i==0:
            previous_cost_slice = cost_slice
            new_assign = tf.zeros(model.N_I)
        else:
            compare = tf.sign(previous_cost_slice - cost_slice)
            compare = tf.maximum(compare, tf.zeros_like(compare)) # N_I of 1s and 0s
            new_assign = new_assign - new_assign*compare + i*compare
            previous_cost_slice = tf.minimum(previous_cost_slice, cost_slice)
    new_assign_temp = tf.transpose(tf.one_hot(tf.cast(new_assign, tf.int32), depth=centroid))
    for i in range(model.N_I):
        c = np.argmax(model.I_assign[:,i])
        if np.sum(new_assign_temp[c])==0 or np.random.randn()<-0.5:
            new_assign = tf.tensor_scatter_nd_update(new_assign, [[i]], [c])
            new_assign_temp = tf.transpose(tf.one_hot(tf.cast(new_assign, tf.int32), depth=centroid))
    new_assign = tf.transpose(tf.one_hot(tf.cast(new_assign, tf.int32), depth=centroid))
    changes = tf.reduce_sum(tf.square(model.I_assign - new_assign))
    if tf.reduce_sum(new_assign)!=model.N_I:
        raise ValueError("Reassign error!")
    model.I_assign.assign(new_assign)
    if tf.reduce_sum(model.I_assign)!=model.N_I:
        raise ValueError("Unrecognized error!")
    else:
        # print("Error check for reassign...passed!", tf.cast(tf.reduce_sum(model.I_assign, 1), tf.int32))
        pass
    if balancing=="nonzero":
        balancing_nonzero(model)
    return changes
def reassign_vanilla(model, centroid, balancing="nonzero"): # from K-MEANS
    assert model.N_C == centroid
    centroid_map_repeat = tf.repeat(tf.expand_dims(model.I_C,axis=2), model.N_I, axis=2)
    I_repeat = tf.repeat(tf.expand_dims(model.I,axis=1), centroid, axis=1)
    cost = tf.reduce_sum(tf.square(centroid_map_repeat - I_repeat), axis=0) # L2 loss shape = (centroid,N_I)
    new_assign = tf.transpose(tf.one_hot(tf.math.argmin(cost, 0), depth=centroid))
    changes = tf.reduce_sum(tf.square(model.I_assign - new_assign))
    model.I_assign.assign(new_assign)
    if balancing == "nonzero":
        balancing_nonzero(model)
    return changes
def reassign_vanilla_1(model, centroid, centroid_1): # for the hierarchical layer
    if model.hierarchy:
        centroid_map_repeat = tf.repeat(tf.expand_dims(model.I_C_1,axis=2), centroid, axis=2)
        I_repeat = tf.repeat(tf.expand_dims(model.I_C,axis=1), centroid_1, axis=1)
        cost = tf.reduce_sum(tf.square(centroid_map_repeat - I_repeat), axis=0) #
        new_assign = tf.transpose(tf.one_hot(tf.math.argmin(cost, 0), depth=centroid_1))
        changes = tf.reduce_sum(tf.square(model.I_assign_1 - new_assign))
        model.I_assign_1.assign(new_assign)
        return changes
    else:
        print("Warning: Model does not have a second layer!")
        return None
def balancing_nonzero(model, k=1): # added feature: can balance to partially grow tree
    distribution = tf.cast(tf.reduce_sum(model.I_assign[0:model.N_C,:], 1), tf.int32)
    if tf.reduce_min(distribution)>0:
        return None
    print("Balancing node: ",end="")
    while tf.reduce_min(distribution)<k:
        biggest = tf.argmax(distribution)
        for i in range(model.N_C):
            if distribution[i]<k:
                print(i, end=" ")
                model.I_C[:,i].assign(model.I_C[:,biggest])
                split = tf.cast(tf.sign(np.random.randn(model.N_I)), tf.float32)
                split_pos = model.I_assign[biggest,:] * split
                split_pos = tf.maximum(split_pos, tf.zeros_like(split_pos))
                split_neg = - model.I_assign[biggest, :] * split
                split_neg = tf.maximum(split_neg, tf.zeros_like(split_neg))
                model.I_assign[biggest, :].assign(split_pos)
                model.I_assign[i,:].assign(model.I_assign[i,:] + split_neg)
                distribution = tf.cast(tf.reduce_sum(model.I_assign[0:model.N_C,:], 1), tf.int32)
                break
        if i==model.N_C-1:
            break
    print("...Finished.")
    if tf.reduce_sum(model.I_assign)!=model.N_I:
        raise ValueError("Unrecognized error!")
    if tf.reduce_min(distribution)<k:
        print("WARNING: Balancing not complete.")
    else:
        print("Error check for balancing...passed!", tf.cast(tf.reduce_sum(model.I_assign, 1), tf.int32))
    return None
def growth(model, A, tf_mask, centroid, threshold,
           mode="data_number", split_mode="PCA"):
    if model.N_C==centroid:
        print("Fully grow.")
        return None
    ############ select centroid to break apart #############
    if mode=="grad_norm": # gradient norm
        with tf.GradientTape() as t:
            current_loss = lossFrobenius(A, model(pred_with_centroid=True), tf_mask) # centroid prediction loss
            """ not consider other losses"""
        dI_C = t.gradient(current_loss, [model.I_C])[0]
        criterion = tf.reduce_sum(tf.square(dI_C), 0)
    elif mode=="number":
        distribution = tf.reduce_sum(model.I_assign, 1)
        criterion = tf.maximum(distribution, tf.ones_like(distribution))
    elif mode=="data_number":
        distribution = tf.reduce_sum(tf.matmul(tf.cast(tf_mask,tf.float32),
                      tf.transpose(model.I_assign)), 0)
        criterion = tf.maximum(distribution, tf.ones_like(distribution))
    elif mode=="loss":
        diff = (tf.cast(A, tf.float32) - model()) * tf_mask
        criterion = tf.reduce_sum(tf.matmul(tf.square(diff), tf.transpose(model.I_assign)), 0)  # [centroid]
    elif mode=="mean_loss":
        diff = (tf.cast(A, tf.float32) - model()) * tf_mask
        criterion = tf.reduce_sum(tf.matmul(tf.square(diff), tf.transpose(model.I_assign)), 0)  # [centroid]
        distribution = tf.reduce_sum(model.I_assign, 1)
        distribution = tf.maximum(distribution, tf.ones_like(distribution))
        criterion = criterion/distribution
    else:
        raise ValueError("Criterion not recognized.")
    chosen_centroid = tf.argmax(criterion)
    if split_mode=="PCA":
        """======================== gradient PCA ========================"""
        with tf.GradientTape() as t: # not consider other losses
            current_loss = lossFrobenius(A, model(), tf_mask)
        dI = t.gradient(current_loss, [model.I])[0]
        reduced_mapping = tf.boolean_mask(tf.linalg.diag(model.I_assign[chosen_centroid]), tf.cast(model.I_assign[chosen_centroid], tf.int32), axis=1)
        dist = PCA_1stVec_split(dI, tf.cast(model.I_assign[chosen_centroid], tf.int32))
        if threshold=="data":
            data_count = []
            for _, i in enumerate(model.I_assign[chosen_centroid]):
                if i == 1:
                    data_count.append(np.sum(tf_mask[:, _]))
            for i in range(len(dist)):
                if np.sum(data_count[0:i])>np.sum(data_count[i:]):
                    med = dist[i]
                    break
            dist = dist - med
        elif threshold=="median":
            median = np.median(dist)
            dist = dist - median
        signvec = tf.sign(dist)
        ############# adding one more centroid ##############
        signvec_positive = tf.sign(signvec + 1)
        signvec_negative = -tf.sign(signvec - 1)
        model.I_assign[chosen_centroid].assign(tf.einsum("a,ba->b",signvec_positive,reduced_mapping))
        model.I_assign[model.N_C].assign(tf.einsum("a,ba->b", signvec_negative, reduced_mapping))
        model.I_C[:,model.N_C].assign(model.I_C[:,chosen_centroid])
        model.N_C += 1
    else:
        raise ValueError("split_mode not recognized.")
