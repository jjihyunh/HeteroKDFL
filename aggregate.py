'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf

class Aggregate:
    def __init__ (self, models):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_model_types = len(models)
        self.num_accumulates = np.zeros((self.num_model_types))
        self.t_params = []
        self.nt_params = []
        for model in models:
            t_weight = []
            nt_weight = []
            for i in range (len(model.trainable_variables)):
                t_weight.append(tf.identity(model.trainable_variables[i]))
            for i in range (len(model.non_trainable_variables)):
                nt_weight.append(tf.identity(model.non_trainable_variables[i]))
            self.t_params.append(t_weight)
            self.nt_params.append(nt_weight)

        self.t_updates = []
        self.nt_updates = []
        for model in models:
            t_update = []
            nt_update = []
            for i in range (len(model.trainable_variables)):
                t_update.append(tf.identity(model.trainable_variables[i]))
            for i in range (len(model.non_trainable_variables)):
                nt_update.append(tf.identity(model.non_trainable_variables[i]))
            self.t_updates.append(t_update)
            self.nt_updates.append(nt_update)

    def init_updates (self, checkpoint, target_model):
        self.num_accumulates[target_model] = 0
        for i in range (len(checkpoint.models[target_model].trainable_variables)):
            self.t_updates[target_model][i] = tf.zeros_like(checkpoint.models[target_model].trainable_variables[i])
        for i in range (len(checkpoint.models[target_model].non_trainable_variables)):
            self.nt_updates[target_model][i] = tf.zeros_like(checkpoint.models[target_model].non_trainable_variables[i])

    def init_model (self, checkpoint, target_model):
        for i in range (len(checkpoint.models[target_model].trainable_variables)):
            checkpoint.models[target_model].trainable_variables[i].assign(self.t_params[target_model][i])
        for i in range (len(checkpoint.models[target_model].non_trainable_variables)):
            checkpoint.models[target_model].non_trainable_variables[i].assign(self.nt_params[target_model][i])

    def accumulate_update (self, checkpoint, target_model):
        norm = 0
        model = checkpoint.models[target_model]
        self.num_accumulates[target_model] += 1
        for i in range (len(model.trainable_variables)):
            delta = tf.math.subtract(model.trainable_variables[i], self.t_params[target_model][i])
            self.t_updates[target_model][i] = tf.math.add(self.t_updates[target_model][i], delta)
            norm += np.linalg.norm(delta.numpy().flatten())**2
        norm = math.sqrt(norm)

        for i in range (len(model.non_trainable_variables)):
            delta = tf.math.subtract(model.non_trainable_variables[i], self.nt_params[target_model][i])
            self.nt_updates[target_model][i] = tf.math.add(self.nt_updates[target_model][i], delta)
        return norm

    def average_model (self, checkpoint, target_model):
        num_accumulates = self.comm.allreduce(self.num_accumulates[target_model], op = MPI.SUM)
        if num_accumulates > 0:
            for i in range (len(checkpoint.models[target_model].trainable_variables)):
                global_update = self.comm.allreduce(self.t_updates[target_model][i], op = MPI.SUM) / num_accumulates
                self.t_params[target_model][i] = tf.math.add(self.t_params[target_model][i], global_update)
            for i in range (len(checkpoint.models[target_model].non_trainable_variables)):
                global_update = self.comm.allreduce(self.nt_updates[target_model][i], op = MPI.SUM) / num_accumulates
                self.nt_params[target_model][i] = tf.math.add(self.nt_params[target_model][i], global_update)
