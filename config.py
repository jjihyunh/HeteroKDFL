'''
Jihyun Lim <wlguslim@inha.edu>
Inha University
2024/10/08
'''
batch_size = 32
lr = 0.2
num_classes = 10
epochs = 400
decay = {200,300}
weight_decay = 0.0001
average_interval = 30
num_workers = 20
checkpoint = 0
active_ratio = 0.2
alpha = 0.1
num_strongs = 20

'''
0: the proposed method
1: baseline(weak)
2: baseline(strong)
'''
optimizer = 2