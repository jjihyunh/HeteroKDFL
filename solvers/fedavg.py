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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

class FedAvg:
    def __init__ (self, num_models, num_classes, num_workers, average_interval):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.average_interval = average_interval
        self.temperature =3
        self.ld =1
        self.local_optimizers = []
        for i in range (num_models):
            self.local_optimizers.append(SGD(momentum = 0.9))
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, checkpoint, dataset, target_model):
        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss = self.train_step(checkpoint, images, labels, target_model)
            lossmean(loss)
        return lossmean

    def distill_round (self, round_id, checkpoint, dataset, public_dataset, target_model):
        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss = self.distill_step(checkpoint,images, labels,0, target_model,round_id) # labeld dataset distillation
            lossmean(loss)
        for i in range(self.average_interval):
            public_images =public_dataset.next()
            pkl = self.distill_step2(checkpoint,public_images,0, target_model,round_id) # unlabeld dataset distillation
        return lossmean

    def train_step (self, checkpoint, data, label, target_model):
        with tf.GradientTape() as tape:
            prediction = checkpoint.models[target_model](data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = checkpoint.models[target_model].losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, checkpoint.models[target_model].trainable_variables)
        checkpoint.optimizers[target_model].apply_gradients(zip(grads, checkpoint.models[target_model].trainable_variables))
        return loss

    def distill_step (self, checkpoint, data, label,teacher, student,round_id):
        with tf.GradientTape() as tape:
            student_logit = checkpoint.models[student](data, training = True)
            student_ce = self.cross_entropy_batch(label, student_logit)
            student_loss = student_ce
            student_regularization_losses = checkpoint.models[student].losses
            student_total_loss = tf.add_n(student_regularization_losses + [student_loss])
        grads = tape.gradient(student_total_loss, checkpoint.models[student].trainable_variables)
        checkpoint.optimizers[student].apply_gradients(zip(grads, checkpoint.models[student].trainable_variables))
        return student_loss

    def distill_step2 (self, checkpoint, data, teacher, student,round_id):
        with tf.GradientTape() as tape:
            teacher_public_logit = tf.keras.activations.softmax(checkpoint.models[teacher](data, training = True) / self.temperature)
            student_public_logit=tf.keras.activations.softmax(checkpoint.models[student](data, training = True) / self.temperature)
            student_public_kd=tf.reduce_mean(tf.keras.losses.KLDivergence()(teacher_public_logit, student_public_logit) * self.temperature**2)
            student_loss= self.ld*student_public_kd
            student_regularization_losses = checkpoint.models[student].losses
            student_total_loss = tf.add_n(student_regularization_losses + [student_loss])
        grads = tape.gradient(student_total_loss, checkpoint.models[student].trainable_variables)
        checkpoint.optimizers[student].apply_gradients(zip(grads, checkpoint.models[student].trainable_variables))
        return student_loss