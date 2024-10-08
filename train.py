'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
import time
import math
import random
import numpy as np
import tensorflow as tf
import argparse
import client_sampling
from mpi4py import MPI
from tqdm import tqdm
from tensorflow.keras.metrics import Mean

class framework:
    def __init__ (self, models, dataset, solver, aggregator, **kargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.dataset = dataset
        self.solver = solver
        self.aggregator = aggregator
        self.models = models
        self.num_epochs = kargs["num_epochs"]
        self.lr = kargs["lr"]
        self.decay_epochs = kargs["decay_epochs"]
        self.average_interval = kargs["average_interval"]
        self.do_checkpoint = kargs["do_checkpoint"]
        self.num_classes = kargs["num_classes"]
        self.num_workers = kargs["num_workers"]
        self.num_clients = kargs["num_clients"]
        self.num_strongs=kargs["num_strongs"]
        self.num_local_workers = int(self.num_workers / self.size)
        self.outter_rounds =1
        self.lr_decay_factor = 10
        self.sampler = client_sampling.sampling(self.num_clients, self.num_workers)
        if self.num_classes == 1:
            self.valid_acc = tf.keras.metrics.BinaryAccuracy()
        else:
            self.valid_acc = tf.keras.metrics.Accuracy()
        self.checkpoint = tf.train.Checkpoint(models = models, optimizers = self.solver.local_optimizers)
        for optimizer in self.checkpoint.optimizers:
            optimizer.lr.assign(self.lr)
        checkpoint_dir = "./checkpoint"
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                              directory = checkpoint_dir,
                                                              max_to_keep = 3)
        # create all the local private datasets.
        self.train_datasets = []
        for i in range (self.num_clients):
            self.train_datasets.append(self.dataset.train_dataset(i))
        # create all the local public datasets.
        self.public_datasets=[]
        for i in range (self.num_strongs):
            self.public_datasets.append(self.dataset.public_dataset(i))

    def train (self):
        # Broadcast the parameters from rank 0 at the first epoch.
        start_epoch = 0
        if start_epoch == 0:
            self.broadcast_model()

        for epoch_id in range (start_epoch, self.num_epochs):
            if epoch_id in self.decay_epochs:
                lr_decay = 1 / self.lr_decay_factor
                for optimizer in self.checkpoint.optimizers:
                    optimizer.lr.assign(optimizer.lr * lr_decay)
            active_devices = self.sampler.random()
            offset = self.num_local_workers * self.rank
            local_devices = active_devices[offset: offset + self.num_local_workers]

            # Collect the active train datasets.
            train_datasets = []
            for i in range (len(local_devices)):
                train_datasets.append(self.train_datasets[local_devices[i]])

            # Training loop.
            target_model = 0
            losses = []
            norms = []
            self.aggregator.init_updates(self.checkpoint, target_model)
            for i in tqdm(range(self.num_local_workers), ascii=True):
                self.aggregator.init_model(self.checkpoint, target_model)
                lossmean = self.solver.round(epoch_id, self.checkpoint, train_datasets[i], target_model)
                norm = self.aggregator.accumulate_update(self.checkpoint, target_model)
                losses.append(lossmean)
                norms.append(norm)

            # Average the model.
            self.aggregator.average_model(self.checkpoint, target_model)

            # Collect the global training results (loss and accuracy).
            local_losses = []
            for i in range (self.num_local_workers):
                local_losses.append(losses[i].result().numpy())
            global_loss = self.comm.allreduce(sum(local_losses), op = MPI.SUM) / self.num_workers

            # Collect the global validation accuracy.
            local_acc = self.evaluate(target_model)
            global_acc = self.comm.allreduce(local_acc, op = MPI.MAX)

            # Checkpointing
            if self.do_checkpoint == True and epoch_id < 250:
                self.checkpoint_manager.save()

            # Logging.
            if self.rank == 0:
                print ("Epoch " + str(epoch_id) +
                    " lr: " + str(self.checkpoint.optimizers[target_model].lr.numpy()) +
                    " validation acc = " + str(global_acc) +
                    " training loss = " + str(global_loss))
                f = open("acc.txt", "a")
                f.write(str(global_acc) + "\n")
                f.close()

                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()

            # Distillation Block
            if len(self.models) > 1: 
                # Find the active strong clients.
                target_model = 1
                num_active_strongs = self.num_strongs
                num_local_active_strongs = num_active_strongs // self.size
                active_devices = np.random.choice(np.arange(self.num_strongs), size = num_active_strongs, replace = False)
                offset = num_local_active_strongs * self.rank
                local_devices = active_devices[offset: offset + num_local_active_strongs]

                # Collect the active public datasets.
                public_datasets=[]
                for i in range(len(local_devices)):
                    public_datasets.append(self.public_datasets[local_devices[i]])

                losses = []
                norms = []
                for outter_round in range (self.outter_rounds):
                    self.aggregator.init_updates(self.checkpoint, target_model)
                    for i in tqdm(range(num_local_active_strongs), ascii=True):
                        self.aggregator.init_model(self.checkpoint, target_model)
                        lossmean = self.solver.distill_round(epoch_id, self.checkpoint, public_datasets[i],target_model)
                        norm = self.aggregator.accumulate_update(self.checkpoint, target_model)
                        losses.append(lossmean)
                        norms.append(norm)
                    # Average the model.
                    self.aggregator.average_model(self.checkpoint, target_model)

                # Collect the global training results (loss and accuracy).
                local_losses = []
                for i in range (num_local_active_strongs):
                    local_losses.append(losses[i].result().numpy())
                global_loss = self.comm.allreduce(sum(local_losses), op = MPI.SUM) / num_active_strongs

                local_acc = self.evaluate(target_model)
                global_acc = self.comm.allreduce(local_acc, op = MPI.MAX)

                # Logging.
                if self.rank == 0:
                    print ("Distillation " + str(epoch_id) +
                        " lr: " + str(self.checkpoint.optimizers[target_model].lr.numpy()) +
                        " validation acc = " + str(global_acc) +
                        " training loss = " + str(global_loss))
                    f = open("dacc.txt", "a")
                    f.write(str(global_acc) + "\n")
                    f.close()

                    f = open("dloss.txt", "a")
                    f.write(str(global_loss) + "\n")
                    f.close()


    def evaluate (self, target_model):
        valid_dataset = self.dataset.valid_dataset()
        self.valid_acc.reset_states()
        self.aggregator.init_model(self.checkpoint,target_model)
        for i in tqdm(range(self.dataset.num_valid_batches), ascii=True):
            data, label = valid_dataset.next()
            predicts = self.checkpoint.models[target_model](data)
            predicts = tf.keras.activations.softmax(predicts)
            if len(label.shape) == 1:
                self.valid_acc(label, predicts)
            else:
                self.valid_acc(tf.argmax(label, 1), tf.argmax(predicts, 1))
        accuracy = self.valid_acc.result().numpy()
        return accuracy

    def broadcast_model (self):
        for i in range(len(self.checkpoint.models)):                                                                                                                                         
            for j in range (len(self.checkpoint.models[i].trainable_variables)):
                param = self.checkpoint.models[i].trainable_variables[j]
                param = self.comm.bcast(param, root=0)
                self.checkpoint.models[i].trainable_variables[j].assign(param)
