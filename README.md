
# RWMFW on Decentralized Online Multi Class Logistic Regression problem.

This project simulates the Random Walk Meta Frank-Wolfe (RWMFW) algorithm for decentralized online optimization of multiclass logistic regression using the MNIST and CIFAR10 datasets.

 # Setup
To set up the environment for the experiment, run the following command:

```bash
./init.sh
```

This will create a `regret` folder where the results of the experiment will be saved.

# Datasets
This project is designed to handle the MNIST and CIFAR-10 datasets. 
To download a preprocessed version of one of these datasets, 
use the following command found in the `dataset` directory:

```bash
./download_preprocessed_dataset.sh mnist
./download_preprocessed_dataset.sh cifar10
```
Make sure to place the downloaded dataset file in the `dataset` directory.

# Configuration
The `config` directory contains configuration files for the experiment:

- `param.conf`: This file contains parameters for the experiment, including the dataset to be used, the batch size, and the number of online rounds, the number of iterations for training.
- `graph.conf`: This file contains information about the type of graph to be used in the experiment and the number of nodes in the graph.


To modify the configuration parameters, you can use the `modify_config.sh` script as follows:

```bash
./modify_config.sh [parameter] [value]
```

- bs: Modifies the batch size.
- l: Modifies the number of iterations for training.
- t: Modifies the number of online rounds.
- trials: Modifies the number of restarts.
- g: Modifies the type and size of the graph to be used in the experiment.

Here are some examples of how to use the modify_config.sh script:

For example, to modify the batch size to 20, you can use the following command:

```bash
./modify_config.sh bs 20
```

To specify the type and size of the graph to be used in the experiment, use the following command:
```bash
./modify_config.sh g [type] [size] [parameters (if needed)]
```

For example, to specify a line graph with 40 nodes, you can use the following command:
```bash
./modify_config.sh g line 40
```

Another example, to specify a grid graph of 5 rows and 10 columns with 50 nodes, you can use the following command:
```bash
./modify_config.sh g grid 50 5 10
```

## Switching Datasets
To switch the dataset for the experiment, you can use the `modify_config.sh` script as follows:

```bash
./modify_config.sh [dataset]
```
The available options for the dataset parameter are:

- mnist: This will change the default parameters for handling the MNIST dataset.
- cifar10: This will change the default parameters for handling the CIFAR10 dataset.

Here is an example of how to switch to the MNIST dataset:

```bash
./modify_config.sh mnist
```
This will update the param.conf file with the following parameters:

```bash
[DATAINFO]
dataset = sorted_mnist.csv
f = 784
c = 10

[ALGOCONFIG]
batch_size = 600
l = 10
t = 100
r = 8
eta = 1
eta_exp = 1
rho = 4
rho_exp = 0.5
reg = 20
Here is an example of how to switch to the CIFAR10 dataset:
```
```bash
./modify_config.sh cifar10
```
This will update the param.conf file with the following parameters:

```bash
[DATAINFO]
dataset = sorted_cifar10.csv
f = 3072
c = 10

[ALGOCONFIG]
batch_size = 500
l = 10
t = 100
r = 32
eta = 0.1
eta_exp = 1
rho = 1
rho_exp = 0.5
reg = 100
```

## Switching Algorithms
By default, the algorithm that is used in the experiment is RWMFW with decentralized settings. If you want to compare with MFW and run the same experiment in centralized settings, you can switch algorithms using the `modify_config.sh` script as follows:

```bash
./modify_config.sh [algorithm]
```
The available options for the algorithm parameter are:

- `rwmfw`: This will run the experiment using the RWMFW algorithm with decentralized settings.
- `mfw` : This will run the experiment using the MFW algorithm with centralized settings.
Here is an example of how to switch to the MFW algorithm:

```bash
./modify_config.sh mfw
```
This will update the `param.conf` file with the following parameter:


```bash
[ALGOCONFIG]
algo = mfw
```

# Running the experiment
To run the experiment with the desired parameters, use the following command:

```bash
./launch_experiment.sh
```
The results of the experiment will be saved in the `regret` folder.