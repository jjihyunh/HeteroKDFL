'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
import tensorflow as tf
import numpy as np

def cos_sim (a, b):
    sum_a = 0
    sum_b = 0
    dot = 0
    for i in range (len(a)):
        sum_a += tf.norm(a[i])**2
        sum_b += tf.norm(a[i])**2
        dot += tf.reduce_sum(tf.math.multiply(a[i], b[i]))
    norm_a = tf.math.sqrt(sum_a)
    norm_b = tf.math.sqrt(sum_b)
    sim = dot / (norm_a * norm_b)
    return sim

def power_d (num_clients, num_workers, weights):
    return np.random.choice(np.arange(num_clients), size = num_workers, replace = False, p=weights)
