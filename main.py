'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
import numpy as np
import tensorflow as tf
import config as cfg
from train import framework
from mpi4py import MPI
from solvers.fedavg import FedAvg
from aggregate import Aggregate
from model import resnet20
from feeders.feeder_cifar import cifar

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    local_rank = rank % len(gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')
        
    # Dataset
    num_clients = int(cfg.num_workers / cfg.active_ratio)
    dataset = cifar(batch_size = cfg.batch_size,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_classes = cfg.num_classes,
                        alpha = cfg.alpha,
                        num_strongs=cfg.num_strongs)
    
    if rank == 0:
        print ("---------------------------------------------------")
        print ("dataset: " + "cifar10")
        print ("number of workers: " + str(cfg.num_workers))
        print ("average interval: " + str(cfg.average_interval))
        print ("batch_size: " + str(cfg.batch_size))
        print ("training epochs: " + str(cfg.epochs))
        print ("---------------------------------------------------")
        
    # Model 
    model = resnet20(cfg.weight_decay, cfg.num_classes, 0.25).build_model()
    model2 = resnet20(cfg.weight_decay, cfg.num_classes, 1).build_model()
    models = [model, model2]
    aggregator = Aggregate(models)
    
    # Fedavg
    solver = FedAvg(num_models = len(models),
                        num_classes = cfg.num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    
    # Training
    trainer = framework(models = models,
                        dataset = dataset,
                        solver = solver,
                        aggregator = aggregator,
                        num_epochs = cfg.epochs,
                        lr = cfg.lr,
                        decay_epochs = cfg.decay,
                        num_classes = cfg.num_classes,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        average_interval = cfg.average_interval,
                        do_checkpoint = cfg.checkpoint,
                        num_strongs= cfg.num_strongs,
                        optimizer = cfg.optimizer)

    trainer.train()