# On-device Distillation Framework
We use CIFAR-10 (Krizhevsky,Hinton et al. 2009) and the dataset is partitioned into two distinct subsets: one for training (private data) and the other for evaluating the model accuracy (public data). 
The strong clients with high system capability have target model(Resnet-20) and auxiliary model(A ResNet-20 model with 25% of the channels at each layer). And the weak clients with relatively insufficient system resources have only auxiliary model.

# Software Requirements ( <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"> <img src="https://img.shields.io/badge/numpy-3776AB?style=for-the-badge&logo=numpy&logoColor=white"> )
To install the required packages for this project, you can use the `requirements.txt` file.  Follow these steps:

* **Step 1: Make sure you have Python and pip installed on your system.**
* **Step 2: Clone this repository:**

   ```bash
   git clone https://github.com/jjihyunh/distillation
* **Step 3:  Install the required packages:**
    ```bash
   pip3 install -r requirements.txt

## Instructions
### Training
 1. Set hyper-parameters properly in `config.py`.
    
      *  **General hyperparameters** :  batch_size,  lr(learning rate),  num_classes,  epochs,  decay(epochs for learning rate decay),  weight_decay,  checkpoint(whether checkpoints are enabled (True) or disabled (False))

      *  **Algorithm-specific hyperparameters** : alpha(dirichlet coefficient),  average_interval(local update steps),  num_workers(total clients participating in Federated Learning),  active_ratio(the proportion of clients randomly selected in each round),  num_strongs(the number of strong clients)

 3. Run training.
    
    The  'number of processes' parameter allows you to take advantage of multiple CPU cores or nodes to execute tasks simultaneously.
      ```
      mpirun -np <number of processes> python3 main.py
      ```
### Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 1. `loss.txt`: An output file that contains the training loss of auxiliary model for every epoch.
 2. `acc.txt`: An output file that contains the validation accuracy of auxiliary model for every epoch.
 3. `dloss.txt`: An output file that contains the training loss of target model for every epoch.
 4. `dacc.txt`: An output file that contains the validation accuracy of target model for every epoch.
 5. `./checkpoint`: The checkpoint files generated after every epoch. This directory is created only when `checkpoint` is set to 1 in `config.py`.
## Questions / Comments
 * Jihyun Lim (wlguslim@inha.edu)
