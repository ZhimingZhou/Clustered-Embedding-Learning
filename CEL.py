import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np

class CEL(tf.Module):
    """ in this model, the centroids are updated with SGD instead of hard assign to center."""
    def __init__(self,
                 initialization="default",
                 structure="growth",
                 base_model="NMF",
                 hierarchy=True,
                 initial_cluster=1,
                 rank=64,
                 reduced_rank=16,
                 N_I=1000,
                 N_U=1000,
                 centroid=100,
                 centroid_1=1,
                 initial_frac_logit=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.hierarchy = hierarchy
        # Initializing random I and U
        self.N_C = initial_cluster
        self.structure = structure
        self.N_I=N_I
        self.N_U=N_U
        temp_I = np.random.randn(rank, N_I).astype(np.float32)
        temp_I = np.divide(temp_I, temp_I.max())
        if initialization == 'abs':
            temp_I = np.abs(temp_I)
        if 'growth' in structure:
            temp_I_assign = np.zeros((centroid, N_I)).astype(np.float32)
            for i in range(N_I):
                c = i % self.N_C # modulo hash as initialization
                temp_I_assign[c,i] += 1
            distribution = np.sum(temp_I_assign,1)
            tf.print("Initialization: ", distribution)
            distribution = np.maximum(np.ones_like(distribution), distribution) # well condition for division
            temp_I_C = np.matmul(np.matmul(temp_I, np.transpose(temp_I_assign)) , np.diag(1/distribution))
            temp_I_C = np.abs(temp_I_C).astype(np.float32) # initialize centroids in positive region
        if self.hierarchy:
            temp_I_assign_1 = np.random.randn(centroid_1, centroid).astype(np.float32)
            temp_I_assign_1 = 1.0 * (temp_I_assign_1.min(axis=0, keepdims=1) == temp_I_assign_1)  # random assign
            distribution_1 = np.sum(temp_I_assign_1, 1)
            for i in range(centroid_1):  # check well condition
                if distribution_1[i] == 0:
                    distribution_1[i] = 1
            print("Initialization_1: ", distribution_1)
            temp_I_C_1 = np.matmul(np.matmul(temp_I_C, np.transpose(temp_I_assign_1)), np.diag(1 / distribution_1))
            temp_I_C_1 = np.abs(temp_I_C_1).astype(np.float32)  # initialize centroids in positive region

        temp_U = np.random.randn(N_U, rank).astype(np.float32)
        temp_U = np.divide(temp_U, temp_U.max())
        if initialization == 'abs':
            temp_U = np.abs(temp_U)
        self.I = tf.Variable(temp_I)
        self.U = tf.Variable(temp_U)
        self.I_C = tf.Variable(temp_I_C)
        self.I_assign = tf.Variable(temp_I_assign.astype(np.float32))
        if self.hierarchy:
            self.I_C_1 = tf.Variable(temp_I_C_1)
            self.I_assign_1 = tf.Variable(temp_I_assign_1.astype(np.float32))
        if self.base_model == "MLP" or self.base_model == "NeuNMF":
            temp_wu1 = 1 / np.sqrt(rank / 2) * np.random.randn(rank, int(rank / 2)).astype(np.float32)
            temp_wu2 = 1 / np.sqrt(1/2 * np.sqrt(reduced_rank * rank / 2)) * np.random.randn(int(rank / 2), reduced_rank).astype(np.float32)
            temp_bu1 = np.zeros([int(rank / 2)]).astype(np.float32)
            temp_bu2 = np.zeros([reduced_rank]).astype(np.float32)
            self.wu1 = tf.Variable(temp_wu1)
            self.wu2 = tf.Variable(temp_wu2)
            self.bu1 = tf.Variable(temp_bu1)
            self.bu2 = tf.Variable(temp_bu2)
            temp_wi1 = 1 / np.sqrt(rank / 2) * np.random.randn(rank, int(rank / 2)).astype(np.float32)
            temp_wi2 = 1 / np.sqrt(1 / 2 * np.sqrt(reduced_rank * rank / 2)) * np.random.randn(int(rank / 2), reduced_rank).astype(np.float32)
            temp_bi1 = np.zeros([int(rank / 2)]).astype(np.float32)
            temp_bi2 = np.zeros([reduced_rank]).astype(np.float32)
            self.wi1 = tf.Variable(temp_wi1)
            self.wi2 = tf.Variable(temp_wi2)
            self.bi1 = tf.Variable(temp_bi1)
            self.bi2 = tf.Variable(temp_bi2)
            if self.base_model == "NeuNMF":
                self.frac = tf.Variable([initial_frac_logit], dtype=tf.float32)
    def __call__(self, pred_with_centroid = False, temp_I = None):
        if temp_I is None:
            if pred_with_centroid:
                temp_I = tf.matmul(self.I_C, self.I_assign)
            else:
                temp_I = self.I
        if self.base_model == "NMF":
            return tf.matmul(self.U, temp_I)
        elif self.base_model == "MLP" or self.base_model == "NeuNMF":
            U_1 = tf.nn.relu(tf.einsum("ab,bc->ac", self.U, self.wu1) + self.bu1)
            U_2 = tf.nn.relu(tf.einsum("ab,bc->ac", U_1, self.wu2) + self.bu2)
            I_1t = tf.nn.relu(tf.einsum("ab,bc->ac", tf.transpose(temp_I), self.wi1) + self.bi1)
            I_2t = tf.nn.relu(tf.einsum("ab,bc->ac", I_1t, self.wi2) + self.bi2)
            input_3 = tf.expand_dims(tf.matmul(U_2,tf.transpose(I_2t)),2)
            if self.base_model == "MLP":
                return tf.squeeze(input_3, 2)
            elif self.base_model == "NeuNMF":
                frac = tf.math.sigmoid(self.frac)
                return frac * tf.squeeze(input_3, 2) + (1-frac) * tf.matmul(self.U, temp_I)
        else:
            raise ValueError("Unknown base model.")